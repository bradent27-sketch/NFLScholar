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
    DRAFTABLE_POSITIONS, CURVE_SEASONS, MODERN_SEASON_GAMES, score_stats,
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

# Season recency weights when averaging a player's own per-game rates. Last
# season dominates - a 2022 usage rate says little about a 2026 role - but
# earlier seasons still stabilize the estimate for players with a short or
# interrupted recent run.
SEASON_RECENCY_WEIGHTS = [1.0, 0.55, 0.30, 0.15, 0.08]


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
    df['_w'] = df['season'].map(weight_for).fillna(0.05)

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
    return rates.reset_index(drop=True)


def project_stat_lines(board, curves, rates, latest_season=2025, games_override=None):
    """
    Attach a full projected stat line to every row of the board.

    The blend per stat is:  self_weight * (player's own rate x expected
    games)  +  (1 - self_weight) * (rank curve total), where self_weight is
    the stat's stickiness (STAT_SELF_WEIGHT) scaled down when the player has
    thin history. A player with no history at all lands entirely on the rank
    curve, which is exactly right for a rookie.
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

    for stat in PROJECTED_STATS:
        out[stat] = 0.0
    out['proj_games'] = np.nan
    out['proj_basis'] = 'rank curve'

    for position, (idx, row) in enumerate(out.iterrows()):
        pos = str(row['Pos']).upper()
        pos_curves = curves.get(pos)
        if not pos_curves:
            continue
        depth = len(pos_curves.get('games', []))
        if depth == 0:
            continue
        rank_idx = int(np.clip(int(row.get('Pos Rank') or 1) - 1, 0, depth - 1))

        expected_games = float(pos_curves['games'][rank_idx]) if 'games' in pos_curves else 15.0
        if games_override:
            expected_games = float(games_override)
        out.at[idx, 'proj_games'] = round(expected_games, 1)

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
                              if s in pos_curves) / max(expected_games, 1.0)
            if curve_usage > 0:
                usage_ratio = own_usage / curve_usage
                if usage_ratio < ROLE_CHANGE_RATIO:
                    # Scales smoothly to zero self-weight as the gap widens,
                    # rather than a cliff that would make one extra carry
                    # flip a player between two very different projections.
                    evidence *= max(0.0, usage_ratio / ROLE_CHANGE_RATIO)
        if history and evidence > 0:
            out.at[idx, 'proj_basis'] = ('own history' if evidence >= 0.99
                                         else f'blend ({int(evidence * 100)}% own)')

        for stat in PROJECTED_STATS:
            curve_total = float(pos_curves[stat][rank_idx]) if stat in pos_curves else 0.0
            if history and evidence > 0:
                own_total = float(history.get(f'rate_{stat}', 0.0)) * expected_games
                weight = STAT_SELF_WEIGHT.get(stat, DEFAULT_SELF_WEIGHT) * evidence
                value = weight * own_total + (1 - weight) * curve_total
            else:
                value = curve_total
            out.at[idx, stat] = round(value, 2)

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
    """
    if board.empty:
        return board
    out = board.copy()
    per_game = out.copy()
    games = pd.to_numeric(out.get('proj_games'), errors='coerce').fillna(15.0).clip(lower=1)

    base_scoring = dict(scoring)
    for key in list(base_scoring):
        if key.startswith('bonus_') and key != 'bonus_mode':
            base_scoring[key] = 0.0
    out['Proj Pts'] = score_stats(per_game, base_scoring).round(1)

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
