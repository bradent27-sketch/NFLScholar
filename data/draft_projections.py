"""
Volume-based statistical projections: a per-player stat line (carries,
targets, yards, touchdowns) that is then scored under your league settings.

WHY THIS REPLACES THE RANK-TO-POINTS CURVE: the previous model went
consensus rank -> "what players finishing at that rank have historically
scored". That works, but it prices scoring settings by analogy rather than
by arithmetic. A 0.5-point-per-carry league or a 100-yard rushing bonus
doesn't shift a points curve in any principled way, because the curve has
already collapsed the stat line into a single number - the carries are gone
by the time the scoring rules arrive.

Projecting the stat line first inverts that. Once a player is 264 carries /
1,350 rushing yards / 14 TDs / 73 receptions, every scoring rule you own is
a multiplication, and the yardage bonuses can be applied to a real
distribution of games rather than approximated. It is also the only way the
board can honestly answer "why" - a projection you can inspect line by line
is one you can disagree with.

HOW EACH STAT IS PROJECTED, and why they are not treated alike:

  1. From this app's local weekly history, build a curve per position per
     stat: what the player finishing Nth at that position actually did.
     Season totals, not per-game, so games missed are already priced in the
     same way the value curves price them.
  2. Take the player's own recent per-game rates, recency-weighted.
  3. Blend the two - and the blend weight DIFFERS BY STAT, because the
     stats differ wildly in how much they carry over year to year. Usage is
     sticky: a back who got 260 carries is likely to get carries again,
     because it's a coaching decision that already happened. Efficiency is
     much less sticky. Touchdown rate is barely sticky at all - it's the
     noisiest thing in fantasy football, and treating a 14-TD season as
     predictive of another one is the single most common way a projection
     goes wrong. So carries and targets lean on the player's own history,
     while touchdown rates get pulled hard toward what's normal for his
     role.

Rookies and role-changers have no usable history, so they fall back
entirely to the rank curve - which is the right answer, since consensus
rank is the only real information about a player who hasn't played.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.draft_board import (
    DRAFTABLE_POSITIONS, CURVE_SEASONS, MODERN_SEASON_GAMES, PACE_GAMES, score_stats,
    expected_games,
)

# The stat line produced for every player. These are exactly the fields
# data.draft_board.score_stats consumes, so a projected line scores through
# the same path as a real one.
PROJECTED_STATS = [
    'carries', 'rushing_yards', 'rushing_tds',
    'targets', 'receptions', 'receiving_yards', 'receiving_tds',
    'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions',
    'rushing_fumbles_lost', 'receiving_fumbles_lost',
    'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39',
    'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_', 'pat_made',
]

# How much of a player's own history to trust, per stat, at full sample.
# The remainder comes from the rank curve.
#
# These are not arbitrary. They encode the best-established regularity in
# fantasy projection: opportunity persists, efficiency regresses, and
# touchdowns regress hardest. A player's carry count is a coaching decision
# that has already been made and is likely to be made again; his touchdown
# total is a handful of goal-line coin flips. Weighting them identically -
# which is what a single blend factor would do - systematically overrates
# whoever just had a fluky scoring season and underrates the workhorse who
# didn't finish drives.
STAT_SELF_WEIGHT = {
    'carries': 0.70, 'targets': 0.70, 'attempts': 0.70,     # usage: sticky
    'receptions': 0.65,
    'rushing_yards': 0.55, 'receiving_yards': 0.55, 'passing_yards': 0.60,
    'rushing_tds': 0.30, 'receiving_tds': 0.30, 'passing_tds': 0.40,
    'passing_interceptions': 0.35,
    'rushing_fumbles_lost': 0.25, 'receiving_fumbles_lost': 0.25,
}
DEFAULT_SELF_WEIGHT = 0.45

# Games of recent history needed before a player's own rates are trusted at
# the full weight above. Below this the blend slides toward the rank curve
# in proportion to how much evidence there actually is.
FULL_TRUST_GAMES = 24

# Below this ratio of a player's own per-game workload to what his consensus
# rank implies, his history is treated as describing a role he no longer has
# (see the role-change damping in project_stat_lines). 0.6 is deliberately
# generous - real usage bounces around year to year, and only a large gap
# should be read as a changed job rather than as noise.
ROLE_CHANGE_RATIO = 0.6

# HOW MANY GAMES A PROJECTION IS SPREAD ACROSS.
#
# The two sides of the blend need a games figure and they need DIFFERENT
# ones, because they already mean different things:
#
#   * The rank curve's entry at rank r is an average season total over every
#     player who finished there. That is already the right unconditional
#     expectation for a player projected at rank r - missed games included,
#     because some of the players in that average missed games. It needs no
#     adjustment at all.
#
#   * The player's own side is a pure per-game rate with no games
#     information in it, so it has to be multiplied by something. That
#     something is the position's EXPECTED STARTER GAMES (see EXPECTED_GAMES
#     in data.draft_board) - not a full season, which was the old answer and
#     was wrong twice over. Weeks 1-17 cannot hold 17 games for a normal
#     player since the schedule went to 18 weeks, and a measured starter
#     plays 13.5-13.9 of the 16 that are available.
#
# Two other candidates were built and measured, and both were worse:
#
#   Rank-curve games ("this player will miss as many games as the average
#   player who finished at his rank"). Not a durability measure - the curve
#   slopes 16.6 at QB1 to 10.6 at QB28 mostly because missing games is how
#   players END UP ranked 28th. It left every QB ~30 points under the analyst
#   consensus (Jared Goff 247 here vs 302 at FFA) purely from the games
#   assumption. Rescaling the curve the other way, up to a flat 17, is worse
#   still: it invents a player who plays a full season and finishes 28th
#   anyway, flattening every positional curve. Measured, that dropped rank
#   agreement with FantasyPros ECR from 0.937 to 0.905 and pushed all four
#   positions 15-28 points ABOVE consensus.
#
#   The player's own games-per-season record. This one sounds right and
#   isn't, because at this sample size it mostly measures role changes and
#   rookie seasons rather than health - a rookie who played 10 games as a
#   backup and 17 as a starter reads as "fragile" at 13.5. Measured, it hurt
#   agreement on both references (ECR rank-corr 0.937 -> 0.924, median
#   positional bias 9.0 -> 11.0) by marking down exactly the players
#   analysts had already cleared, Malik Nabers and Rashee Rice among them.
#
# What separates the measured EXPECTED_GAMES from that rejected third option
# is WHOSE record is used. Projecting a player on his own games history
# measures his role changes and rookie seasons; projecting him on the
# positional average for players who held a starting role measures
# availability, which is the thing actually being estimated. One is a
# player-specific durability claim this data cannot support, the other is a
# base rate that it can.
#
# PACE_GAMES and EXPECTED_GAMES both live in data.draft_board, because the
# valuation side needs the same constants.

# Season recency weights when averaging a player's own per-game rates. Last
# season dominates - a 2022 usage rate says little about a 2026 role - but
# earlier seasons still stabilize the estimate for players with a short or
# interrupted recent run.
SEASON_RECENCY_WEIGHTS = [1.0, 0.55, 0.30, 0.15, 0.08]

# SEASON COMPLETENESS. Recency answers "how relevant is this season";
# completeness answers a different question, "how much of this player did
# we actually see" - and a season cut short by injury scores low on the
# second one no matter how high it scores on the first. Before this, an
# 8-game injury season carried the same PER-GAME weight as a full 17-game
# one at the same recency slot, so a shortened season sitting in the most
# recent slot (weight 1.0) could swing the blend as hard as a full season
# would, and quality seasons just one slot further back faded fast under
# SEASON_RECENCY_WEIGHTS regardless of how good they were. Confirmed real:
# Joe Burrow's 8-game 2025 (reduced pace, turf toe) sat at recency weight
# 1.0 while his healthy, better 2024 (17 games, 289 yd/g, 43 TD) had
# already faded to 0.55 - the blend read closer to the hurt season than to
# his actual current form.
#
# Scaled toward SEASON_COMPLETENESS_FLOOR rather than to zero: a short
# season is still real evidence, just noisier and less representative than
# a full one. Dropping it out entirely would trade one distortion for
# another - a rookie who played 6 committee games would vanish from his own
# projection instead of just counting for less.
SEASON_COMPLETENESS_FLOOR = 0.45

# AGING. Measured here rather than assumed, off 1,644 contributor seasons
# (>=8 games, >=4 ppg) from 2014 on, matched to birthdates from the
# DynastyProcess ID crosswalk. The quantity measured is the median
# year-over-year ratio of points per game within each age band, and only the
# SHAPE across bands is used - the level is meaningless because requiring a
# productive season N selects high, so every band's ratio sits below 1.
#
#   RB  22:0.98  24:0.84  26:0.85  28:0.73  30:0.68
#   WR  22:1.01  24:0.91  26:0.85  28:0.78  30:0.76  32:0.67
#   TE  22:0.98  24:1.07  26:0.79  28:0.83  30:0.80  32:0.90
#   QB  22:1.02  24:0.96  26:0.93  28:0.92  30:0.93  32:0.90  36:0.91
#
# Backs fall off a cliff and receivers slide steadily; tight ends and
# quarterbacks essentially don't, which is the well-known shape and a good
# sign the measurement isn't an artifact. TE and QB rates below come from
# the regression that controls for prior-year production rather than from
# these raw bands, because their raw bands are flat-but-noisy and the
# regression separates the small real decline from the noise.
#
# Rates are per year of age past the position's peak, as a fraction of the
# per-game rate. The first entry of each list below IS that position's peak;
# the age enters no other way, which is why there is no separate peak table.
#
# These rates STEEPEN WITH AGE. They used to be a single flat rate per
# position (QB 0.010, RB 0.065, WR 0.036, TE 0.020), which described the
# first year or two past the peak and nothing beyond it.
#
# WHAT WAS WRONG WITH THAT, and it was worse than "too gentle": because the
# markdown is integrated only over the interval a player's history has to be
# aged forward (HISTORY_LAG_SEASONS), a constant rate makes the multiplier
# CONSTANT for everyone past the peak. Measured on the live board, a
# 27-year-old back and a 37-year-old back both came out at 0.883 - identical.
# The curve stopped doing anything at all after about 27, which is precisely
# where it should start mattering, and it is why the audit found the board
# buying 30+ veterans (Kamara, Hill, Diggs, Hockenson) rounds ahead of the
# market.
#
# MEASURED from 1,072 year-over-year pairs in local history: how much of his
# own per-game production a player kept the following season, by age. Read
# against each position's peak band, since even young players show some
# regression to the mean:
#
#   RB   26-28: 0.86   28-30: 0.73   30-32: 0.64
#   WR   26-28: 0.84   28-30: 0.76   30-32: 0.72   32+: 0.81
#   TE   26-28: 0.94   28-30: 0.77   30-32: 0.99   32+: 0.87
#   QB   26-28: 0.93   28-30: 0.97   30-32: 0.92   32+: 0.89
#
# The sample is survivorship-biased UPWARD - it can only include players who
# were still playing six games in both seasons, so the ones who fell off a
# cliff are missing - which means the true decline is steeper than these
# numbers, not gentler. The rates below are still set a little inside the
# measurement, because the older bands are thin (RB 30-32 is n=14) and
# because the market and the consensus rank already carry some of this.
#
# TIGHT ENDS DELIBERATELY GET LITTLE. Their bands are non-monotone (0.77 then
# 0.99) on small samples, so the data does not support an aggressive TE
# cliff, and inventing one to make Kelce and Kittle look right would be
# fitting the model to two players. Their gap to the market is better
# explained by injury history, which is handled separately below.
AGE_DECLINE_BANDS = {
    'RB': [(25.0, 0.065), (28.0, 0.120), (30.0, 0.190)],
    'WR': [(26.0, 0.036), (28.0, 0.090), (30.0, 0.130)],
    'TE': [(27.0, 0.020), (30.0, 0.055)],
    'QB': [(27.0, 0.010), (34.0, 0.040)],
}

# A steeper band past 34 was built and measured and is deliberately ABSENT.
# It is neutral on every metric (ECR rank-corr 0.962 -> 0.963, projection MAE
# 19.8 both ways, TE MAE 12.3 -> 12.2) and it rests on almost nothing: past
# 34 the local history holds 12 running back seasons, 26 receiver, 23 tight
# end. Adding a parameter that moves nothing measurable, fitted to a dozen
# players, is how a model starts encoding its author's hunches. The market
# blend already handles the genuinely ancient outliers.

# How far in the past a player's own rates sit, in seasons, relative to the
# season being projected. It is the weight-centroid of SEASON_RECENCY_WEIGHTS
# (0.92 seasons back) plus the one season forward being projected. This is
# the interval the decline is integrated over - a player's history is not
# one year stale, and pretending it is would under-apply the curve.
HISTORY_LAG_SEASONS = 1.92

# Applied only as a markdown, never as a boost, even though the raw bands do
# show young players gaining (~4.5%/yr at RB/WR/TE, ~2% at QB relative to
# prime). The symmetric version was built and measured, and it does nothing:
# rank agreement with FantasyPros ECR and with FFA Value were identical to
# three decimals, and projection MAE got marginally worse (22.0 -> 22.2).
# That is not a surprise in hindsight - young players are exactly the ones
# with thin history, so `evidence` is already low and the consensus rank
# curve is already carrying most of their projection. Boosting the small
# own-history slice barely moves them. The upside half is left to the market,
# which prices it better anyway.
#
# The floor was 0.75, which was itself masking the flat-curve bug: with the
# old constant rates nothing ever reached it, and had the rates been steepened
# without lowering it, a 34-year-old back and a 29-year-old would have come
# out identical again for a different reason. 0.45 is set below anything the
# bands can reach in a single lag interval, so it now guards only against a
# nonsense age rather than capping real decline.
AGE_ADJUST_MIN = 0.45


def _weekly_history(latest_season, n_seasons=CURVE_SEASONS):
    from data.loaders import load_weekly_stats_history
    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'season' not in hist.columns:
        return pd.DataFrame(), None
    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    if name_col not in df.columns or 'position' not in df.columns:
        return pd.DataFrame(), None
    seasons = sorted([s for s in df['season'].dropna().unique() if s <= latest_season])[-n_seasons:]
    return df[df['season'].isin(seasons)].copy(), name_col


def _stat_series(df, stat):
    """Stat column with alias handling, else zeros."""
    aliases = {'carries': ['carries', 'rushing_attempts'],
               'attempts': ['attempts', 'passing_attempts'],
               'passing_interceptions': ['passing_interceptions', 'interceptions']}
    for candidate in aliases.get(stat, [stat]):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors='coerce').fillna(0)
    return pd.Series(0.0, index=df.index)


@st.cache_data(show_spinner=False)
def build_volume_curves(latest_season, n_seasons=CURVE_SEASONS, ppr_for_ranking=1.0):
    """
    curves[pos][stat][i] = season total of that stat for the player who
    finished (i+1)th at the position, averaged over recent seasons. Plus a
    'games' entry, so a player's own per-game rates can be scaled to a
    realistic games-played figure rather than a flat 17.

    Ranked by PPR points regardless of the user's actual scoring: this curve
    describes typical usage at each rung of a position's pecking order, and
    that ordering shouldn't shuffle every time someone toggles a scoring
    setting. The user's real scoring is applied later, to the projected stat
    line, which is the whole point of doing it this way.
    """
    df, name_col = _weekly_history(latest_season, n_seasons)
    if df.empty:
        return {}

    ranking_scoring = {
        'pass_yd': 0.04, 'pass_td': 4.0, 'pass_int': -2.0, 'pass_2pt': 2.0,
        'rush_yd': 0.1, 'rush_td': 6.0, 'rush_att': 0.0, 'rush_2pt': 2.0,
        'rec': float(ppr_for_ranking), 'rec_yd': 0.1, 'rec_td': 6.0, 'rec_2pt': 2.0,
        'te_premium': 0.0, 'fumble_lost': -2.0,
        'fg_0_39': 3.0, 'fg_40_49': 4.0, 'fg_50_plus': 5.0, 'pat': 1.0,
        'bonus_mode': 'cumulative',
    }
    df['_rank_points'] = score_stats(df, ranking_scoring)

    agg = {stat: (stat, 'sum') for stat in PROJECTED_STATS if stat in df.columns}
    for stat in PROJECTED_STATS:
        if stat not in df.columns:
            df[stat] = _stat_series(df, stat)
            agg[stat] = (stat, 'sum')
    agg['_rank_points'] = ('_rank_points', 'sum')
    agg['games'] = ('week', 'nunique')
    agg['position'] = ('position', 'first')

    totals = df.groupby(['season', name_col], observed=True).agg(**agg).reset_index()

    curves = {}
    for pos in DRAFTABLE_POSITIONS:
        pos_rows = totals[totals['position'].astype(str).str.upper() == pos]
        if pos_rows.empty:
            continue
        per_season = {}
        depth = 0
        for season in sorted(pos_rows['season'].unique()):
            season_rows = pos_rows[pos_rows['season'] == season].sort_values('_rank_points', ascending=False)
            if len(season_rows) < 10:
                continue
            scale = MODERN_SEASON_GAMES / 16.0 if season < 2021 else 1.0
            per_season[season] = (season_rows, scale)
            depth = max(depth, len(season_rows))
        if not per_season:
            continue

        pos_curves = {}
        for stat in PROJECTED_STATS + ['games']:
            stacked = []
            for season_rows, scale in per_season.values():
                if stat not in season_rows.columns:
                    continue
                values = pd.to_numeric(season_rows[stat], errors='coerce').fillna(0).to_numpy(dtype=float)
                # 'games' is a count of weeks, not a rate - scaling a
                # 16-game season's games-played up to 17 would invent a game
                # nobody played, so only the volume stats get rescaled.
                if stat != 'games':
                    values = values * scale
                if len(values) < depth:
                    values = np.concatenate([values, np.full(depth - len(values), values[-1] if len(values) else 0.0)])
                stacked.append(values[:depth])
            if stacked:
                pos_curves[stat] = np.vstack(stacked).mean(axis=0)
        if pos_curves:
            curves[pos] = pos_curves
    return curves


@st.cache_data(show_spinner=False)
def build_player_rates(latest_season, n_seasons=CURVE_SEASONS):
    """
    Each player's recency-weighted per-game rate for every projected stat,
    plus how many games of evidence sit behind it.

    Per-game rather than per-season because a player who missed half a year
    to injury still tells you what his role was when he played, and that
    role is what carries into next season. The games count travels alongside
    so the blend downstream can tell a 40-game sample from a 3-game one.
    """
    df, name_col = _weekly_history(latest_season, n_seasons)
    if df.empty:
        return pd.DataFrame()

    seasons = sorted(df['season'].unique(), reverse=True)
    weight_for = {s: (SEASON_RECENCY_WEIGHTS[i] if i < len(SEASON_RECENCY_WEIGHTS) else 0.05)
                  for i, s in enumerate(seasons)}
    recency_weight = df['season'].map(weight_for).fillna(0.05)

    # Completeness scales EACH row's weight by how much of that player's
    # OWN season it came from - a season with 8 games contributes 8 rows at
    # a discounted per-row weight, on top of already contributing fewer rows
    # than a full season would. Both effects are real and distinct: fewer
    # rows is "less data from this season"; the discount is "and what's
    # there is a less reliable sample of him", which matters most exactly
    # when a short season also happens to be the most-recent, highest-weight
    # one.
    games_in_season = df.groupby([name_col, 'season'])['week'].transform('nunique')
    completeness = (games_in_season / MODERN_SEASON_GAMES).clip(upper=1.0)
    completeness_factor = SEASON_COMPLETENESS_FLOOR + (1 - SEASON_COMPLETENESS_FLOOR) * completeness
    df['_w'] = (recency_weight * completeness_factor).fillna(0.05)

    for stat in PROJECTED_STATS:
        df[stat] = _stat_series(df, stat)

    # Vectorized weighted mean rather than a per-player Python loop: this
    # runs over ~130k weekly rows for ~3,700 players, and the loop version
    # dominated the whole projection build.
    weighted = pd.DataFrame({f'w_{stat}': df[stat] * df['_w'] for stat in PROJECTED_STATS})
    weighted[name_col] = df[name_col].values
    weighted['position'] = df['position'].values
    weighted['_w'] = df['_w'].values

    grouped = weighted.groupby([name_col, 'position'], observed=True)
    sums = grouped[[f'w_{stat}' for stat in PROJECTED_STATS]].sum()
    weight_totals = grouped['_w'].sum()
    games_sample = grouped.size().rename('games_sample')

    rates = sums.div(weight_totals, axis=0)
    rates.columns = [c.replace('w_', 'rate_') for c in rates.columns]
    rates = rates.join(games_sample).reset_index()
    rates = rates.rename(columns={name_col: 'Player', 'position': 'Pos'})
    rates['Pos'] = rates['Pos'].astype(str).str.upper()
    rates = rates[weight_totals.reset_index(drop=True).values > 0] if len(rates) else rates
    if rates.empty:
        return rates
    # One row per player: a player who changed listed position across
    # seasons would otherwise appear twice and get half his sample each.
    rates = rates.sort_values('games_sample', ascending=False).drop_duplicates('Player', keep='first')

    # Games played in the MOST RECENT season only, carried alongside the
    # rates. Distinct from games_sample, which spans five years and measures
    # how much evidence there is; this measures whether he was healthy last
    # year, which is a different question and the one injury_adjustment
    # answers.
    last = df[df['season'] == max(seasons)]
    played = last.groupby(name_col).size().rename('games_last_season')
    rates = rates.merge(played.reset_index().rename(columns={name_col: 'Player'}),
                        on='Player', how='left')
    return rates.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_age_lookup(season):
    """
    Player -> age (in years) at kickoff of the given season.

    Computed from BIRTHDATE rather than read off the crosswalk's own `age`
    column, because a birthdate is a fact and an age is a fact with a
    timestamp on it - the published column is correct only for whenever that
    file was last rebuilt. The two agree to 0.27 years where both exist,
    which is the drift you'd expect, and the birthdate version can't rot.

    Keyed on the loose merge name because this joins to a consensus feed
    that spells suffixes differently, and an age is harmless to get slightly
    wrong on a rare collision - it scales a projection by a few percent, it
    doesn't decide anything on its own.
    """
    from data.draft_sources import load_player_id_map
    from data.utils import clean_name_for_merge

    ids, _ = load_player_id_map()
    if ids is None or ids.empty or 'birthdate' not in ids.columns:
        return {}
    ids = ids.dropna(subset=['birthdate']).copy()
    born = pd.to_datetime(ids['birthdate'], errors='coerce')
    ids = ids[born.notna()]
    born = born[born.notna()]
    kickoff = pd.Timestamp(f'{int(season)}-09-01')
    ages = ((kickoff - born).dt.days / 365.25).round(2)
    keys = clean_name_for_merge(ids['name'])
    lookup = {}
    for key, age in zip(keys, ages):
        if key and 18 <= age <= 50:
            lookup.setdefault(key, float(age))

    # NICKNAME TIER. The two keys above normalize punctuation and suffixes,
    # which is not the failure that actually happens here: the crosswalk
    # carries legal names and the consensus feed carries the name on the
    # jersey, so "Kenneth Gainwell" and "Kenny Gainwell" never meet. He was
    # one of the missing ages, and missing means NO aging markdown at all -
    # a 27-year-old back silently priced as though age were unknown.
    #
    # Keyed on last name + the first THREE letters of the first name +
    # position. Every part is load-bearing, and the first-initial version
    # this replaced proves it: on real data it recovered the four genuine
    # nicknames (Kenneth/Kenny, Andres/Andy, Chigoziem/Chig,
    # Mitchell/Mitch) and also four complete strangers - Devonte Boyd
    # matched a retired kicker named Danny Boyd and came back 48 years old.
    # A shared first initial is not evidence. Three letters separates every
    # real nickname here from every false one, because nicknames are
    # overwhelmingly truncations, and position stops a receiver from
    # inheriting a linebacker's birthday.
    #
    # Ambiguous keys are dropped rather than guessed: a wrong age quietly
    # scales a projection, which is exactly the kind of error nobody would
    # ever go looking for. Stored under a '~' prefix so it can only be
    # consulted after both exact tiers have missed.
    positions = ids['position'].astype(str).str.upper() if 'position' in ids.columns else None
    seen, ambiguous = {}, set()
    for i, (name, age) in enumerate(zip(ids['name'].astype(str), ages)):
        pos = positions.iloc[i] if positions is not None else ''
        nickname_key = _nickname_key(name, pos)
        # 40 rather than 50: this tier exists to find ACTIVE players, and
        # the crosswalk is full of retirees whose only claim is a surname.
        if not nickname_key or not (18 <= age <= 40):
            continue
        if nickname_key in seen and abs(seen[nickname_key] - float(age)) > 0.05:
            ambiguous.add(nickname_key)
        seen.setdefault(nickname_key, float(age))
    for key, age in seen.items():
        if key not in ambiguous:
            lookup[f'~{key}'] = age
    return lookup


# The crosswalk spells kickers PK; the board says K. Nothing else differs
# across the positions this board drafts.
_CROSSWALK_POSITION_ALIASES = {'PK': 'K'}


def _nickname_key(name, position=''):
    """
    ("Kenneth Gainwell", "RB") -> "gainwell|ken|RB".

    Last name, the first three letters of the first name, and position -
    the parts of a name that survive a nickname without also matching a
    stranger. Suffixes are dropped so "Odell Beckham Jr." keys on beckham,
    and anything that isn't at least two words returns None, since a
    single-token name has no last name to key on and would collide with
    everything.
    """
    import re
    parts = [p for p in re.sub(r'[^a-z ]', '', str(name).lower()).split() if p]
    while len(parts) > 2 and parts[-1] in ('jr', 'sr', 'ii', 'iii', 'iv', 'v'):
        parts.pop()
    if len(parts) < 2 or len(parts[0]) < 3:
        return None
    pos = str(position or '').upper()
    pos = _CROSSWALK_POSITION_ALIASES.get(pos, pos)
    return f"{parts[-1]}|{parts[0][:3]}|{pos}"


def age_adjustment(position, age):
    """
    Multiplier on a player's own historical per-game rates, for the fact
    that he will be older next season than he was when he produced them.

    Only ever <= 1: see AGE_ADJUST_MIN for why the upside half is left to
    the consensus rank rather than modelled here. Returns 1.0 for an unknown
    age, an unknown position, or anyone at or below his position's peak, so
    a missing birthdate costs nothing.

    The decline is integrated over the interval between where his history
    actually sits (HISTORY_LAG_SEASONS back) and the season being projected,
    counting only the part of that interval past the peak. A 26-year-old
    back is therefore charged for the one year of it that lands past 25,
    not the full lag, which keeps the curve continuous at the peak instead
    of stepping down the moment a player has a birthday.
    """
    pos = str(position).upper()
    bands = AGE_DECLINE_BANDS.get(pos)
    if not bands or age is None or not np.isfinite(age):
        return 1.0

    # Integrate the (now age-dependent) rate across the interval his history
    # has to be carried forward, counting only the part past the peak. Doing
    # it as an integral rather than a lookup keeps the curve continuous - a
    # player doesn't step down the day he crosses a band boundary, he pays
    # the steeper rate only for the months of the interval that sit past it.
    start = float(age) - HISTORY_LAG_SEASONS
    end = float(age)
    exponent = 0.0
    for index, (band_from, rate) in enumerate(bands):
        band_to = bands[index + 1][0] if index + 1 < len(bands) else float('inf')
        overlap = min(end, band_to) - max(start, band_from)
        if overlap > 0:
            exponent += rate * overlap
    if exponent <= 0:
        return 1.0
    return float(max(np.exp(-exponent), AGE_ADJUST_MIN))


# WHAT LAST SEASON'S MISSED GAMES SAY ABOUT NEXT SEASON, as
# (games played last year, per-game multiplier, games multiplier).
#
# MEASURED over the same 1,072 year-over-year pairs as the aging bands. A
# player's next season, by how much of the previous one he played:
#
#   played 15-17   n=570   next-year games 13.7   per-game retained 0.92
#   played 12-14   n=300   next-year games 13.1   per-game retained 0.88
#   played  8-11   n=174   next-year games 12.3   per-game retained 0.79
#   played   1-7   n= 28   next-year games 13.1   per-game retained 0.83
#
# Both halves move, which is the point: a player coming off a serious injury
# is likelier to miss time AGAIN and likelier to be worse per game when he
# plays. Expressed relative to the healthy band, so a fully-fit player is
# untouched and nobody is marked down merely for existing.
#
# The 1-7 band is treated as no better than 8-11 despite measuring slightly
# higher. It is 28 pairs, and it is the most survivorship-biased cell in the
# table - a player who lost almost all of a season only appears here if he
# came back and played six games the next year, which is precisely the subset
# that recovered.
#
# THIS IS NOT THE SAME IDEA AS "project his own games-per-season", which was
# tried before and measurably hurt (see the games-basis note above): that
# used a five-year games average as the projection basis and mostly detected
# rookies and role changes. This keys on last season only, applies as a
# markdown rather than as the basis, and is validated the same way - against
# rank agreement with FantasyPros and FFA.
INJURY_HISTORY_BANDS = [
    (15, 1.00, 1.00),
    (12, 0.97, 0.97),
    (8, 0.88, 0.92),
    (0, 0.88, 0.90),
]


# Positions the injury markdown applies to. QUARTERBACK IS EXCLUDED, and the
# measurement is unambiguous about why: applying it to QBs moved their
# projection error the wrong way (MAE 50.6 -> 53.2, bias -10.0 -> -17.2)
# while improving every other position. The reason is that "games played" is
# a far worse proxy for health at quarterback than anywhere else - a QB who
# played eight games usually lost his JOB rather than got hurt, and marking
# him down for it double-counts a role change the rank curve has already
# priced. Receivers were borderline (14.1 -> 14.4) and are excluded on the
# same logic, since a part-season for a receiver is very often a rookie
# easing into a role.
#
# Backs and tight ends are where it pays: RB MAE 17.4 -> 16.4, TE 13.2 ->
# 12.3, and TE bias 6.4 -> 3.9. Those are also the positions where a lost
# season is most often a real structural injury.
INJURY_ADJUSTED_POSITIONS = {'RB', 'TE'}

# WORKLOAD IS DELIBERATELY NOT MODELLED, and this note exists so the idea
# isn't re-implemented on intuition later.
#
# "A back with 300 carries breaks down the next year" is one of the most
# repeated claims in fantasy football. It was tested here against the same
# 1,072 year-over-year pairs, holding age under 28 so the effect wasn't just
# aging in disguise, and the data says the opposite:
#
#   RB, by touches (carries + targets) the previous season
#     <150      n=71   next-year games 13.1   per-game retained 0.77
#     150-225   n=72   next-year games 13.5   per-game retained 0.87
#     225-300   n=69   next-year games 14.1   per-game retained 0.94
#     300+      n=33   next-year games 13.5   per-game retained 0.87
#
#   WR/TE, by targets
#     <80       n=202  next-year games 13.0   per-game retained 0.87
#     80-120    n=162  next-year games 13.6   per-game retained 0.91
#     120-160   n=66   next-year games 14.2   per-game retained 0.95
#     160+      n=10   next-year games 14.2   per-game retained 0.88
#
# Heavy usage predicts MORE games played and BETTER retention, not less, at
# every band up to 300 touches, and even the 300+ cell sits above the
# low-usage one. The signal is real but it runs the other way: touches are a
# proxy for job security, and the players with few of them are committee
# backs and rotational receivers whose roles are what's actually fragile.
# A workload penalty would mark down exactly the workhorses you want and
# promote the replacement-level players you don't.
#
# The volume model already captures the useful half of this, since a player's
# own carry and target rates are the stickiest inputs it has.


# FREE AGENCY. A player with no NFL team is not a discounted version of a
# starter - he is a bet that someone signs him, and then a second bet about
# what role he gets. The board had neither bet priced: it read his last
# season's usage rates off local history exactly as it would for a starter,
# and Tyreek Hill came out at board rank 151 against an ADP of 224 and an
# ECR of 244. Every unsigned player on the board sat 60-120 picks ahead of
# where the market had him, in the same direction, which is a model gap
# rather than an edge.
#
# MEASURED on 1,682 pairs: players who were real contributors in season N-1
# (6+ games, 4+ ppg), split by whether they were on an active week-1 roster
# in season N.
#
#                              n     played at all   games   per-game kept
#   on an active roster      1361        99.9%        12.5       0.926
#   NOT on one               321         53.6%         3.6       0.364
#   ...of those, if he played              -           6.7       0.679
#
# Both halves of the user's intuition are in that table and both are large.
# Nearly half never play a down. Those who do play about half a season, and
# at roughly three-quarters of their old per-game rate - a committee back or
# a fourth receiver, which is what a team signing you in September has open.
#
# THE RATES BELOW ARE DELIBERATELY MILDER THAN THE MEASUREMENT (0.62/0.55
# against a measured 0.39/0.29), for a reason that is about reference class
# rather than timidity. The measurement's cutoff is week 1; this board is
# built in August. A player unsigned on the eve of the season has been
# passed over by all 32 teams with the deadline in sight, which is a much
# worse signal than being unsigned in early August with a month of camp
# injuries still to come. Applying a week-1 number to an August free agent
# would price a certainty that hasn't happened yet.
FREE_AGENT_RATE_MULT = 0.62
FREE_AGENT_GAMES_MULT = 0.55

# What the Team column says when a player has no team. The board's Team
# field comes from the same FantasyPros export as the ranks, so this needs
# no name matching to work.
#
# A CROSS-CHECK AGAINST THE 2026 ROSTER FILE WAS BUILT AND REJECTED. Marking
# anyone absent from roster_weekly_2026.csv as a free agent looks more
# authoritative - it's real NFL roster data rather than one vendor's string -
# but it flagged 109 players against FantasyPros' 34, and the extras were
# Patrick Mahomes, James Cook and Travis Etienne. The cause is name
# suffixes: the crosswalk carries "Patrick Mahomes II" and the roster file
# carries "Patrick Mahomes", so clean_name_exact produces two different
# keys. A free-agent penalty that occasionally lands on the best quarterback
# in football is far worse than one that misses a fringe tight end, so the
# signal that needs no matching wins.
FREE_AGENT_TEAM_CODES = {'FA', 'FA*', '', 'NAN', 'NONE', '-', '--', 'N/A'}


def is_free_agent(team):
    """True when this player has no NFL team."""
    return str(team).strip().upper() in FREE_AGENT_TEAM_CODES


def free_agent_adjustment(team):
    """
    (per-game multiplier, games multiplier) for having no team.

    Same two-part shape as injury_adjustment, because the failure is the
    same shape: fewer games, and a smaller role in the games he does play.
    """
    if not is_free_agent(team):
        return 1.0, 1.0
    return FREE_AGENT_RATE_MULT, FREE_AGENT_GAMES_MULT


def injury_adjustment(games_last_season, position=None):
    """
    (per-game multiplier, games multiplier) for how much of last season a
    player actually played. Returns (1.0, 1.0) when it's unknown - a rookie
    or anyone absent from local history is not penalised for having no
    record - and for positions where the measurement said it doesn't help.
    """
    if games_last_season is None or not np.isfinite(games_last_season):
        return 1.0, 1.0
    if position is not None and str(position).upper() not in INJURY_ADJUSTED_POSITIONS:
        return 1.0, 1.0
    played = float(games_last_season)
    for threshold, rate_mult, games_mult in INJURY_HISTORY_BANDS:
        if played >= threshold:
            return rate_mult, games_mult
    return 1.0, 1.0


def project_stat_lines(board, curves, rates, latest_season=2025, ages=None):
    """
    Attach a full projected stat line to every row of the board.

    The blend per stat is:  self_weight * (player's own rate x PACE_GAMES)
    + (1 - self_weight) * (rank curve total), where self_weight is the
    stat's stickiness (STAT_SELF_WEIGHT) scaled down when the player has
    thin history. A player with no history at all lands entirely on the rank
    curve, which is exactly right for a rookie.

    See the games note above PACE_GAMES for why the own side is projected
    across a full season while the curve side is left alone, and for the two
    alternatives that were measured and rejected.

    The own-history side is additionally scaled by age_adjustment(), and
    ONLY that side: the rank-curve side is indexed by consensus rank, and
    consensus has already priced the player's age. Applying an age curve to
    both halves would charge a 33-year-old for his age twice.
    """
    if board.empty or not curves:
        return board

    out = board.copy()
    # Two-tier name matching against the app's own history, not a raw string
    # join. The consensus feed and nflverse disagree on suffixes and
    # punctuation constantly - "James Cook III" vs "James Cook", "Travis
    # Etienne Jr." vs "Travis Etienne" - and an exact join silently drops
    # those players onto the rank curve as though they were rookies with no
    # NFL history at all. Verified: it was doing exactly that to a projected
    # RB6 and RB18.
    from data.utils import clean_name_exact, clean_name_for_merge
    exact_lookup, loose_lookup = {}, {}
    if rates is not None and not rates.empty:
        keys_exact = clean_name_exact(rates['Player'])
        keys_loose = clean_name_for_merge(rates['Player'])
        records = rates.to_dict('records')
        for key, record in zip(keys_exact, records):
            exact_lookup.setdefault(key, record)
        for key, record in zip(keys_loose, records):
            loose_lookup.setdefault(key, record)

    board_exact = clean_name_exact(out['Player'])
    board_loose = clean_name_for_merge(out['Player'])

    def _history_for(position):
        record = exact_lookup.get(board_exact.iloc[position])
        if record is None:
            record = loose_lookup.get(board_loose.iloc[position])
        return record

    ages = ages if ages is not None else build_age_lookup(int(latest_season) + 1)
    # A team defense has no age, and "Kansas City Chiefs" happens to collide
    # with a real person in the crosswalk - which was putting a 47-year-old
    # DST on the board.
    board_ages = [None if str(pos).upper() == 'DST' else
                  (ages.get(k) if ages.get(k) is not None
                   else ages.get(f'~{_nickname_key(name, pos)}'))
                  for k, name, pos in zip(board_loose, out['Player'].astype(str),
                                          out['Pos'].astype(str))]

    for stat in PROJECTED_STATS:
        out[stat] = 0.0
    out['proj_games'] = np.nan
    out['proj_basis'] = 'rank curve'
    out['Age'] = [a if a is not None else np.nan for a in board_ages]
    out['age_factor'] = 1.0
    out['injury_factor'] = 1.0
    out['games_last_season'] = np.nan
    out['free_agent'] = False

    for position, (idx, row) in enumerate(out.iterrows()):
        pos = str(row['Pos']).upper()
        pos_curves = curves.get(pos)
        if not pos_curves:
            continue
        depth = len(pos_curves.get('games', []))
        if depth == 0:
            continue
        # A MISSING Pos Rank must fall to the WORST slot on the curve
        # (depth - 1), never the best. `row.get('Pos Rank') or 1` (the
        # previous version of this line) defaulted a missing/zero rank to
        # 1 - the RB1/WR1 curve, ~380+ fantasy points - which is backwards:
        # a player with no real consensus rank is the LEAST information
        # this board has about him, not evidence he's elite. Confirmed real
        # on a live board: several deep-bench/rookie names with a
        # 0-or-missing ECR/ADP (see data.draft_sources._rank_or_unranked's
        # docstring for where that 0 comes from) were rendering as the
        # board's #1 overall values off this exact fallback.
        raw_pos_rank = row.get('Pos Rank')
        if raw_pos_rank is None or pd.isna(raw_pos_rank) or raw_pos_rank < 1:
            rank_idx = depth - 1
        else:
            rank_idx = int(np.clip(int(raw_pos_rank) - 1, 0, depth - 1))

        # The curve's games figure is used ONLY to turn its season totals
        # back into per-game rates for the role-change comparison below,
        # which is the one place it's the right number: it's asking "what
        # workload did the player who finished here actually carry per game
        # he played". It is deliberately not used as this player's games.
        curve_games = float(pos_curves['games'][rank_idx]) if 'games' in pos_curves else 15.0

        history = _history_for(position)
        sample_games = float(history['games_sample']) if history else 0.0
        evidence = min(1.0, sample_games / FULL_TRUST_GAMES) if sample_games else 0.0

        # ROLE-CHANGE DAMPING. A player's own history is only evidence about
        # next season if he's doing the same job next season. When consensus
        # ranks someone far above anything his usage has ever supported, the
        # market is telling you the job changed - a backup quarterback who
        # just won a starting role, a back whose committee partner left.
        # Blending in his bench-usage rates then drags the projection toward
        # a role he no longer has.
        #
        # Caught by comparing his own per-game workload against what the
        # curve says is normal at his consensus rank. Validated against a
        # professional projection set: overall agreement was already high
        # (r=0.93 on points, r=0.96 on carries), and essentially every large
        # disagreement was exactly this case - backup QBs projected as
        # starters by the analysts and as backups here.
        if history and evidence > 0:
            own_usage = sum(float(history.get(f'rate_{s}', 0.0))
                            for s in ('carries', 'targets', 'attempts'))
            curve_usage = sum(float(pos_curves[s][rank_idx]) for s in ('carries', 'targets', 'attempts')
                              if s in pos_curves) / max(curve_games, 1.0)
            if curve_usage > 0:
                usage_ratio = own_usage / curve_usage
                if usage_ratio < ROLE_CHANGE_RATIO:
                    # Scales smoothly to zero self-weight as the gap widens,
                    # rather than a cliff that would make one extra carry
                    # flip a player between two very different projections.
                    evidence *= max(0.0, usage_ratio / ROLE_CHANGE_RATIO)

                # RECENCY EVIDENCE. games_sample/FULL_TRUST_GAMES alone counts
                # CAREER games, so a player with one clean, uncontested season
                # reads identically to one with the same games total scattered
                # thin across three years of committee snaps - it can't tell
                # "just proved it" from "never proved it." A player who played
                # nearly every week of the season just finished, at or above
                # the workload his current rank implies, has already answered
                # the role-stability question a second season would otherwise
                # be collecting evidence for.
                #
                # Confirmed real: Cam Ward (TEN) started all 17 games as a
                # rookie at 97% of the per-game workload a player at his
                # consensus rank normally carries, and still landed at 71%
                # trust (17 career games / 24) under games_sample alone -
                # diluting a proven, uncontested season toward a curve where
                # half the historical field at that rank didn't hold the job
                # the whole year.
                #
                # Takes the MAX with the evidence above rather than replacing
                # it, so this can only ADD confidence for a confirmed recent
                # season - never undo the role-change guard just above. A
                # real backup granted a full season on the depth chart but
                # not in touches scores low here too, since recency_evidence
                # is scaled by the same usage_ratio that guard uses.
                games_last_season = history.get('games_last_season')
                # NOT `if games_last_season:` - that's truthy for float('nan')
                # (NaN is non-zero), so a player with no games-last-season
                # figure at all was silently read as having played a full
                # 17-game season. Confirmed real: every deep backup QB with
                # an undefined games_last_season and a small, noisy own-rate
                # sample (Sam Ehlinger, Case Keenum, Bailey Zappe...) jumped
                # straight to 100% own-history trust off garbage-time attempt
                # rates, because min(1.0, nan) silently returns 1.0 in
                # Python rather than raising or propagating the NaN.
                if games_last_season is not None and np.isfinite(games_last_season):
                    # Full real-calendar weeks (17), not expected_games(pos) -
                    # this is asking "how much of the schedule did he just
                    # play", a distinct question from "how many games does a
                    # typical starter play" (which already prices in average
                    # missed time and would let this fire too easily).
                    season_share = min(1.0, float(games_last_season) / 17.0)
                    recency_evidence = season_share * min(1.0, max(0.0, usage_ratio))
                    evidence = max(evidence, recency_evidence)
        if history and evidence > 0:
            out.at[idx, 'proj_basis'] = ('own history' if evidence >= 0.99
                                         else f'blend ({int(evidence * 100)}% own)')

        age_factor = age_adjustment(pos, board_ages[position])
        out.at[idx, 'age_factor'] = round(age_factor, 3)

        # Coming off a season that was cut short is its own risk, separate
        # from age: likelier to miss time again, and worse per game when
        # playing. Applied only to the OWN-HISTORY side of the blend and
        # scaled by how much of the projection that side is carrying, because
        # the rank curve already averages over players who missed games -
        # marking it down too would charge the same injury twice.
        injury_rate_mult, injury_games_mult = injury_adjustment(
            history.get('games_last_season') if history else None, pos)
        out.at[idx, 'injury_factor'] = round(injury_rate_mult, 3)
        if history and history.get('games_last_season') is not None:
            out.at[idx, 'games_last_season'] = history.get('games_last_season')

        # Having no team at all, folded in the same way and for the same
        # reason: it is the own-history side that assumes last season's role
        # carries forward, and that assumption is exactly what free agency
        # breaks. The curve side is left alone because a free agent's
        # consensus rank has already been marked down by the market - taking
        # it off both sides would charge his unemployment twice.
        fa_rate_mult, fa_games_mult = free_agent_adjustment(row.get('Team'))
        out.at[idx, 'free_agent'] = fa_rate_mult < 1.0
        injury_rate_mult *= fa_rate_mult
        injury_games_mult *= fa_games_mult

        # The games this line is spread across, which the milestone-bonus
        # model needs to turn a season total back into a weekly mean. It has
        # to follow the blend: the own-history side carries a full season,
        # the curve side carries whatever the players at that rank actually
        # played, so the line as a whole sits between them in proportion to
        # how much of it came from each. Exact at both ends - a pure rookie
        # gets the curve's games, a fully-established player gets the
        # position's expected starter total.
        #
        # THAT EXPECTATION IS NOW POSITION-SPECIFIC AND MEASURED (see
        # EXPECTED_GAMES in data/draft_board.py) rather than a flat full
        # season. The old basis was 17 games, which weeks 1-17 cannot even
        # contain, and it left the top 60 players projected for 16.81 games
        # against a measured 13.5-13.9 for players who held a starting role.
        pace = expected_games(pos)
        games_mult = 1.0 - evidence * (1.0 - injury_games_mult)
        out.at[idx, 'proj_games'] = round(
            (evidence * pace + (1 - evidence) * curve_games) * games_mult, 1)

        for stat in PROJECTED_STATS:
            curve_total = float(pos_curves[stat][rank_idx]) if stat in pos_curves else 0.0
            if history and evidence > 0:
                own_total = (float(history.get(f'rate_{stat}', 0.0)) * pace
                             * age_factor * injury_rate_mult * games_mult)
                weight = STAT_SELF_WEIGHT.get(stat, DEFAULT_SELF_WEIGHT) * evidence
                value = weight * own_total + (1 - weight) * curve_total
            else:
                value = curve_total
            out.at[idx, stat] = round(value, 2)

    # A readable form of last season's availability. A veteran whose
    # projection came down should be able to say why on the row itself -
    # "11/17" beside a lower number explains itself in a way a hidden
    # multiplier never does. Free agency rides in the same column for the
    # same reason: it is the single biggest thing suppressing an unsigned
    # player's number, and the Team column alone doesn't connect the two.
    played = pd.to_numeric(out['games_last_season'], errors='coerce')
    health = np.where(played.notna(),
                      played.fillna(0).astype(int).astype(str) + '/17', '—')
    free = out['free_agent'].fillna(False).astype(bool).to_numpy()
    out['Health'] = np.where(free, np.where(played.notna(), health + ' · FA', 'FA'), health)
    return out


def score_projected_lines(board, scoring):
    """
    Points from the projected stat line under this league's scoring.

    Yardage milestone bonuses need per-GAME yardage, and a projected stat
    line is a season total - so the bonus contribution is estimated by
    modelling each player's weekly yardage as a gamma distribution with his
    projected per-game mean and a position-typical spread, then integrating
    the thresholds over it. Applying the thresholds to the season total
    instead would award a 100-yard bonus once for a 1,400-yard season, and
    ignoring them would drop the setting entirely; both are worse than an
    explicit distributional estimate.

    Divided by the player's own projected games, not by 17: the bonus is
    per GAME PLAYED, so a back who is projected for 1,200 yards across 14
    games clears 100 in a week far more often than one who takes 17 to get
    there, and dividing both by a flat 17 would price them identically.
    """
    if board.empty:
        return board
    out = board.copy()
    per_game = out.copy()
    games = pd.to_numeric(out.get('proj_games'), errors='coerce').fillna(PACE_GAMES).clip(lower=1)

    base_scoring = dict(scoring)
    for key in list(base_scoring):
        if key.startswith('bonus_') and key != 'bonus_mode':
            base_scoring[key] = 0.0
    # Floored at zero. A projected line is a season-long EXPECTATION, and no
    # draftable player has a negative one: the only way this went below zero
    # was a deep bench body whose rank-curve share of fumbles and picks
    # outweighed a near-zero share of the production, which is an artifact of
    # slicing a curve thin rather than a real forecast. Left unclamped it put
    # 20 players on the board at negative points.
    out['Proj Pts'] = score_stats(per_game, base_scoring).clip(lower=0.0).round(1)

    from data.draft_board import YARDAGE_BONUSES, has_yardage_bonuses
    if has_yardage_bonuses(scoring):
        bonus_total = pd.Series(0.0, index=out.index)
        highest_only = str(scoring.get('bonus_mode', 'cumulative')).lower().startswith('high')
        by_stat = {}
        for stat, threshold, key in YARDAGE_BONUSES:
            by_stat.setdefault(stat, []).append((threshold, key))
        for stat, tiers in by_stat.items():
            if stat not in out.columns:
                continue
            season_total = pd.to_numeric(out[stat], errors='coerce').fillna(0)
            mean_per_game = season_total / games
            prev_prob = None
            for threshold, key in sorted(tiers, key=lambda t: -t[0]):
                points = float(scoring.get(key, 0) or 0)
                prob = _prob_game_exceeds(mean_per_game, threshold, stat)
                if points:
                    # 'highest only' pays a threshold just for the games that
                    # cleared it but NOT the one above, which is the
                    # difference of the two exceedance probabilities.
                    effective = prob if (prev_prob is None or not highest_only) else (prob - prev_prob).clip(lower=0)
                    bonus_total = bonus_total + effective * games * points
                prev_prob = prob if prev_prob is None else np.maximum(prev_prob, prob)
        out['Proj Pts'] = (out['Proj Pts'] + bonus_total).round(1)
        out['Bonus Pts'] = bonus_total.round(1)
    return out


# Week-to-week variability of a player's yardage, as a coefficient of
# variation. Rushing is the steadiest (carries arrive whether or not the
# offense is working); receiving swings more; passing yardage sits between.
# Used only to price milestone bonuses, which depend on the SHAPE of a
# player's weekly distribution and not just its mean - two backs with equal
# season yardage earn very different bonus totals if one is a metronome and
# the other alternates 30 and 170.
_YARDAGE_CV = {'rushing_yards': 0.62, 'receiving_yards': 0.78, 'passing_yards': 0.35}


def _prob_game_exceeds(mean_per_game, threshold, stat):
    """
    P(a single game clears `threshold` yards), modelling weekly yardage as a
    gamma with the player's projected mean and a position-typical spread.

    Gamma rather than normal because weekly yardage is non-negative and
    right-skewed - the occasional 180-yard game is exactly what earns these
    bonuses, and a symmetric distribution would systematically under-price
    them for boom/bust players while over-pricing steady ones.
    """
    mean = pd.to_numeric(mean_per_game, errors='coerce').fillna(0).clip(lower=0.01)
    cv = _YARDAGE_CV.get(stat, 0.7)
    shape = 1.0 / (cv ** 2)
    scale = mean / shape
    try:
        from math import lgamma
        # Regularized upper incomplete gamma via a series/continued-fraction
        # pair (Numerical Recipes' gammq), vectorized over players. Avoids a
        # scipy dependency for one function - see data.draft_sources for the
        # same tradeoff on the normal survival function.
        x = np.asarray(threshold / scale, dtype=float)
        a = float(shape)
        return pd.Series(_gammq(a, x), index=mean.index)
    except Exception:
        return pd.Series(0.0, index=mean.index)


def _gammq(a, x, iters=200, eps=1e-9):
    """Regularized upper incomplete gamma Q(a, x), elementwise."""
    from math import lgamma, exp, log
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    gln = lgamma(a)
    for i, xi in enumerate(x):
        if xi <= 0:
            out[i] = 1.0
            continue
        if xi < a + 1.0:
            # Series representation for P(a, x), then Q = 1 - P.
            ap, total, delta = a, 1.0 / a, 1.0 / a
            for _ in range(iters):
                ap += 1.0
                delta *= xi / ap
                total += delta
                if abs(delta) < abs(total) * eps:
                    break
            out[i] = 1.0 - total * exp(-xi + a * log(xi) - gln)
        else:
            # Continued fraction for Q(a, x) directly.
            tiny = 1e-300
            b = xi + 1.0 - a
            c = 1.0 / tiny
            d = 1.0 / b
            h = d
            for n in range(1, iters + 1):
                an = -n * (n - a)
                b += 2.0
                d = an * d + b
                if abs(d) < tiny:
                    d = tiny
                c = b + an / c
                if abs(c) < tiny:
                    c = tiny
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            out[i] = h * exp(-xi + a * log(xi) - gln)
    return np.clip(out, 0.0, 1.0)


def build_projected_board(ecr_board, scoring, latest_season=2025, n_seasons=CURVE_SEASONS,
                          ppr_for_ranking=1.0):
    """
    The full volume-projection pipeline: usage curves, player rates, blended
    stat lines, then scoring. Returns (board_with_stat_line, meta).
    """
    curves = build_volume_curves(latest_season, n_seasons, ppr_for_ranking=ppr_for_ranking)
    if not curves:
        return ecr_board, {'volume_projections': False}
    rates = build_player_rates(latest_season, n_seasons)
    board = project_stat_lines(ecr_board, curves, rates, latest_season=latest_season)
    board = score_projected_lines(board, scoring)

    # Team defenses have no player stat line to project - they don't appear
    # in player-level data at all - so they come through the volume path with
    # zero points, which then drags their replacement level to zero and makes
    # every DST look infinitely valuable. They keep the points-curve
    # projection built from real team-week defensive stats instead. Same
    # fallback covers any other position the volume curves don't reach.
    from data.draft_board import build_positional_value_curves, project_points, calibrate_rank_uncertainty
    missing = [pos for pos in board['Pos'].astype(str).str.upper().unique()
               if pos not in curves]
    if missing:
        point_curves = build_positional_value_curves(scoring, latest_season)
        usable = {pos: point_curves[pos] for pos in missing if pos in point_curves}
        if usable:
            calibration = calibrate_rank_uncertainty(scoring, latest_season)
            patch = project_points(board, usable, calibration=calibration)
            fill = board['Pos'].astype(str).str.upper().isin(usable.keys())
            board.loc[fill, 'Proj Pts'] = patch.loc[fill, 'Proj Pts']
            board.loc[fill, 'proj_basis'] = 'points curve'

    meta = {
        'volume_projections': True,
        'positions_with_curves': sorted(curves.keys()),
        'positions_from_points_curve': sorted(missing),
        'players_with_history': int(len(rates)) if rates is not None else 0,
    }
    return board, meta
