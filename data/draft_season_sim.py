"""
Simulate the season your roster would actually play, and report WINS.

WHY THIS EXISTS. Every other number on this board optimises projected season
points. Nobody wins a league on season points - you win seventeen separate
head-to-head games, and how your points are distributed across those weeks
matters nearly as much as how many you have. Two rosters projecting the same
total are not the same team if one scores 110 every week and the other
alternates 80 and 140: against a typical opponent the steady one wins more
matchups, and wins are what the standings pay out on.

That distinction was invisible here. "Best starting lineup projects 2,077
points, 78th percentile" is a real answer to the wrong question.

HOW IT WORKS

  1. Each player gets a weekly score DISTRIBUTION rather than a per-game
     average, bootstrapped from the real weeks of players who finished at his
     projected rank (data/draft_weekly.py), then rescaled so its mean matches
     THIS board's projection for him. Shape from history, level from the
     model - resampling real weeks keeps the right skew and the fat upper
     tail that a normal distribution around a mean would erase, and those are
     precisely the weeks that win matchups.
  2. Each week every player is independently available or not: out on his bye,
     and otherwise absent at his position's measured rate.
  3. The best legal lineup is fielded from whoever is available - so bench
     depth earns its keep exactly when it should, in the weeks a starter is
     out, rather than through an abstract multiplier.
  4. Teams are paired off each week and the higher score wins.
  5. Repeat a few hundred times for a distribution of records.

WHAT IT IS NOT. This is not a claim to know who wins your league. The inputs
carry every uncertainty the projections do, and the opponent field in a live
draft is reconstructed rather than observed. It is a way to compare two
rosters on the axis the league is actually scored on, and to see that a
high-floor team and a boom-bust team with equal projections are different
bets.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.draft_board import (
    DRAFTABLE_POSITIONS, FLEX_ELIGIBLE, SUPERFLEX_ELIGIBLE, MODERN_SEASON_GAMES,
    measure_absence_rates,
)
from data.draft_weekly import build_weekly_pools

# Draws per player-week come from a pool of a few hundred real weeks; below
# this many samples the tail is one player's good month rather than a
# distribution, so a modelled fallback is used instead.
MIN_POOL = 25

# Spread of the fallback distribution, as a share of the mean, for players
# with no usable pool (deep bench, kickers in odd formats, team defenses).
# 0.55 is roughly the observed weekly coefficient of variation across skill
# positions, so a fallback player behaves like a typical one rather than
# like a metronome.
FALLBACK_CV = 0.55


def _roster_matrix(roster, board_lookup, pools, absence, rng, n_sims, n_weeks):
    """
    Weekly scores for one roster: an (n_sims, n_weeks, n_players) array with
    NaN wherever a player is unavailable that week.

    Built per roster rather than per player so the whole season for a team is
    one vectorised draw - a 12-team league over 200 seasons is 40,000 lineup
    decisions, and doing that a player at a time in Python is the difference
    between a second and a minute.
    """
    n_players = len(roster)
    scores = np.zeros((n_sims, n_weeks, n_players), dtype=float)
    positions = []
    priority = np.zeros(n_players, dtype=float)

    for index, player in enumerate(roster):
        pos = str(player.get('Pos', '')).upper()
        positions.append(pos)
        info = board_lookup.get(player.get('Player')) or {}

        per_game = info.get('per_game')
        if per_game is None or not np.isfinite(per_game):
            season = player.get('Proj Pts')
            per_game = (float(season) / MODERN_SEASON_GAMES
                        if season is not None and pd.notna(season) else 0.0)
        per_game = max(0.0, float(per_game))
        # The lineup priority: what you'd expect from him, which is what a
        # manager sets a lineup on.
        priority[index] = per_game

        pool = pools.get((pos, int(info.get('rank', 0) or 0)))
        if pool is not None and len(pool) >= MIN_POOL and pool.mean() > 0:
            # Shape from real weeks, level from this board's projection.
            draws = rng.choice(pool, size=(n_sims, n_weeks), replace=True)
            scores[:, :, index] = draws * (per_game / float(pool.mean()))
        elif per_game > 0:
            # Gamma, not a clipped normal. Fantasy weeks can't be negative,
            # and clipping a normal at zero quietly ADDS to its mean - by
            # more the wider the spread, so a volatile player would come out
            # projecting for more points than the board gave him and the
            # simulator would reward variance for a purely arithmetic reason.
            # Gamma is non-negative by construction, mean-preserving at any
            # spread, and right-skewed like real weekly scoring.
            shape = 1.0 / (FALLBACK_CV ** 2)
            scores[:, :, index] = rng.gamma(shape, per_game / shape,
                                            size=(n_sims, n_weeks))

        # Unavailability. The bye is certain and known; the rest is the
        # position's measured absence rate, drawn independently per week.
        rate = float(np.clip(absence.get(pos, 0.2), 0.0, 0.6))
        out = rng.random((n_sims, n_weeks)) < rate
        bye = player.get('Bye')
        if bye is not None and pd.notna(bye):
            week = int(bye) - 1
            if 0 <= week < n_weeks:
                out[:, week] = True
        scores[:, :, index] = np.where(out, np.nan, scores[:, :, index])

    return scores, np.array(positions), priority


def _take_best_available(available, priority_order, slots):
    """
    Mask of who fills `slots` spots, choosing by a FIXED priority among
    whoever is available. `available` is (n_sims, n_weeks, n) over the
    columns in `priority_order`, best first.

    Done with a cumulative count rather than a sort because the choice is not
    score-dependent: a player starts if he is available and fewer than `slots`
    higher-priority team-mates also are.
    """
    if slots <= 0 or available.shape[2] == 0:
        return np.zeros_like(available, dtype=bool)
    ordered = available[:, :, priority_order]
    ahead = np.cumsum(ordered, axis=2) - ordered
    chosen_ordered = ordered & (ahead < slots)
    chosen = np.zeros_like(chosen_ordered)
    chosen[:, :, priority_order] = chosen_ordered
    return chosen


def _weekly_lineup_points(scores, positions, roster_cfg, priority=None):
    """
    The lineup a manager would actually have SET each week, scored on what
    then happened. Returns (n_sims, n_weeks).

    THE LINEUP IS CHOSEN ON PROJECTIONS, NOT ON REALISED SCORES, and the
    difference is not a detail. Picking each week's flex by what he ended up
    scoring gives every team perfect hindsight it will never have on Sunday
    morning: it inflates every roster's points, and it inflates volatile
    players most, because a boom-bust flex looks wonderful if you can always
    start him in his boom weeks. Measured on one fixed roster, hindsight
    lineups added 105 points a season at ordinary volatility and 191 at high
    volatility - turning a genuine COST of variance into a spurious reward
    for it, the exact opposite of what this module exists to measure. With
    lineups set on projections the same roster scores the same 1,784 points
    at every volatility level, and the steadier team wins 52-55% of weeks
    against an equally-projected volatile one.

    So the priority order is fixed up front (each player's projected points
    per game), availability varies week to week, and the best AVAILABLE
    players by that fixed order are fielded. Greedy dedicated-then-flex is
    still valid because flex eligibility is nested.
    """
    n_sims, n_weeks, n_players = scores.shape
    if priority is None:
        priority = np.nan_to_num(np.nanmean(scores, axis=(0, 1)), nan=0.0)
    available = ~np.isnan(scores)
    points = np.nan_to_num(scores, nan=0.0)

    started = np.zeros((n_sims, n_weeks, n_players), dtype=bool)
    for pos in DRAFTABLE_POSITIONS:
        columns = np.where(positions == pos)[0]
        slots = int(roster_cfg.get(pos, 0))
        if len(columns) == 0 or slots <= 0:
            continue
        order = np.argsort(-priority[columns])
        chosen = _take_best_available(available[:, :, columns], order, slots)
        started[:, :, columns] |= chosen

    for slot_name, eligible in (('FLEX', FLEX_ELIGIBLE), ('SUPERFLEX', SUPERFLEX_ELIGIBLE)):
        count = int(roster_cfg.get(slot_name, 0))
        if count <= 0:
            continue
        columns = np.where(np.isin(positions, list(eligible)))[0]
        if len(columns) == 0:
            continue
        spare = available[:, :, columns] & ~started[:, :, columns]
        order = np.argsort(-priority[columns])
        chosen = _take_best_available(spare, order, count)
        started[:, :, columns] |= chosen

    return (points * started).sum(axis=2)


def _board_lookup(board):
    """
    Player -> the two things the simulator needs from the board: projected
    points per GAME, and projected finish rank at his position (the key into
    the weekly pools).

    Per game rather than per season because the simulator decides
    availability itself; handing it a season total would apply the games a
    player is expected to miss twice.
    """
    if board is None or board.empty:
        return {}
    work = board.copy()
    pos_upper = work['Pos'].astype(str).str.upper()
    work['_rank'] = work.groupby(pos_upper)['Proj Pts'].rank(ascending=False, method='first')
    games = pd.to_numeric(work.get('proj_games'), errors='coerce') if 'proj_games' in work.columns else None
    out = {}
    for i in range(len(work)):
        row = work.iloc[i]
        season = pd.to_numeric(row.get('Proj Pts'), errors='coerce')
        played = float(games.iloc[i]) if games is not None and pd.notna(games.iloc[i]) else np.nan
        if not np.isfinite(played) or played <= 0:
            played = MODERN_SEASON_GAMES
        out[row['Player']] = {
            'per_game': float(season) / played if pd.notna(season) else np.nan,
            'rank': int(row['_rank']) if pd.notna(row['_rank']) else 0,
        }
    return out


def build_field_rosters(board, settings, my_roster, my_slot=1):
    """
    A plausible set of opponent rosters for a LIVE draft, where the other
    teams are unobserved.

    Assembled by snake-drafting the board's own order over whoever your roster
    hasn't claimed. It is not what your league will actually do, and it isn't
    trying to be - it's a field of competently-drafted teams to measure
    against, which is the right control for "is my roster good". In a mock
    the real simulated rosters are used instead and none of this applies.
    """
    n_teams = int(settings['num_teams'])
    rounds = sum(int(settings['roster'].get(k, 0)) for k in
                 ['QB', 'RB', 'WR', 'TE', 'K', 'DST', 'FLEX', 'SUPERFLEX', 'BENCH'])
    rounds = max(rounds, 1)

    mine = {p.get('Player') for p in (my_roster or [])}
    pool = board[~board['Player'].isin(mine)].copy()
    if 'Board Rank' in pool.columns:
        pool = pool.sort_values('Board Rank')
    pool = pool.head(n_teams * rounds)

    from data.draft_sim import _legal_positions
    rosters = {team: [] for team in range(1, n_teams + 1) if team != my_slot}
    if not rosters:
        return {}
    counts = {team: {} for team in rosters}
    taken = set()
    order = list(rosters.keys())

    for round_no in range(1, rounds + 1):
        sequence = order if round_no % 2 else list(reversed(order))
        for team in sequence:
            legal = _legal_positions(counts[team], settings, round_no, rounds,
                                     rounds - len(rosters[team]))
            candidates = pool[(~pool['Player'].isin(taken)) &
                              (pool['Pos'].astype(str).str.upper().isin(legal))]
            if candidates.empty:
                candidates = pool[~pool['Player'].isin(taken)]
            if candidates.empty:
                continue
            pick = candidates.iloc[0]
            taken.add(pick['Player'])
            entry = {'Player': pick['Player'], 'Pos': str(pick['Pos']).upper(),
                     'Proj Pts': pick.get('Proj Pts'), 'Bye': pick.get('Bye')}
            rosters[team].append(entry)
            pos = entry['Pos']
            counts[team][pos] = counts[team].get(pos, 0) + 1
    return rosters


def simulate_seasons(rosters, board, settings, n_sims=200, seed=17,
                     n_weeks=MODERN_SEASON_GAMES, playoff_teams=None):
    """
    Play the league `n_sims` times. Returns a DataFrame indexed by team with
    mean wins, the spread of records, mean points and playoff odds.

    The schedule is redrawn each simulated season rather than fixed. A real
    league's schedule is one arbitrary draw, and holding it constant would
    bake its luck - who happens to face the best team twice - into every
    answer. Averaging over schedules measures the roster, which is the only
    part of this you control on draft day.
    """
    teams = [team for team, roster in rosters.items() if roster]
    if len(teams) < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    lookup = _board_lookup(board)
    pools, _ = build_weekly_pools(settings['scoring'], settings.get('baseline_season', 2025))
    absence = measure_absence_rates(settings.get('baseline_season', 2025))
    roster_cfg = settings['roster']

    weekly = {}
    for team in teams:
        scores, positions, priority = _roster_matrix(rosters[team], lookup, pools,
                                                     absence, rng, n_sims, n_weeks)
        weekly[team] = _weekly_lineup_points(scores, positions, roster_cfg, priority)

    stacked = np.stack([weekly[t] for t in teams], axis=2)      # (sims, weeks, teams)
    n_teams = len(teams)
    wins = np.zeros((n_sims, n_teams), dtype=float)

    # One fresh random pairing per simulated week. With an odd team count the
    # unpaired team takes a bye that week rather than a free win, which would
    # otherwise be worth more than playing.
    for sim in range(n_sims):
        for week in range(n_weeks):
            order = rng.permutation(n_teams)
            for i in range(0, n_teams - 1, 2):
                a, b = order[i], order[i + 1]
                if stacked[sim, week, a] > stacked[sim, week, b]:
                    wins[sim, a] += 1
                elif stacked[sim, week, b] > stacked[sim, week, a]:
                    wins[sim, b] += 1
                else:
                    wins[sim, a] += 0.5
                    wins[sim, b] += 0.5

    if playoff_teams is None:
        playoff_teams = max(2, n_teams // 2)
    points = stacked.sum(axis=1)                                 # (sims, teams)
    # Ties on record are broken by points scored, which is what nearly every
    # platform does.
    ranking = np.lexsort((-points, -wins), axis=1)
    made = np.zeros((n_sims, n_teams), dtype=bool)
    for sim in range(n_sims):
        made[sim, ranking[sim, :playoff_teams]] = True

    return pd.DataFrame({
        'Team': teams,
        'Wins': wins.mean(axis=0).round(2),
        'Wins P25': np.percentile(wins, 25, axis=0).round(1),
        'Wins P75': np.percentile(wins, 75, axis=0).round(1),
        'Points': points.mean(axis=0).round(0),
        'Playoff %': (made.mean(axis=0) * 100).round(1),
        # Best regular-season record, NOT a championship - a title needs a
        # playoff bracket this doesn't simulate, and calling it one would
        # overclaim by a wide margin.
        'Top Seed %': (np.array([(ranking[s, 0] == t) for s in range(n_sims)
                                 for t in range(n_teams)]).reshape(n_sims, n_teams)
                       .mean(axis=0) * 100).round(1),
    }).set_index('Team')


@st.cache_data(show_spinner=False, ttl=900)
def grade_roster_wins(_board, _settings, roster_signature, my_slot, n_sims=150):
    """
    Expected wins and playoff odds for YOUR roster against a reconstructed
    field, cached on a signature of who you've drafted.

    Cached on the signature rather than the frames for the reason documented
    in data.transforms: hashing a 700-row board on every widget interaction
    costs more than the simulation it would be protecting.
    """
    roster = [{'Player': name, 'Pos': pos, 'Proj Pts': points, 'Bye': bye}
              for name, pos, points, bye in roster_signature]
    if not roster:
        return None
    field = build_field_rosters(_board, _settings, roster, my_slot=my_slot)
    if not field:
        return None
    rosters = dict(field)
    rosters[my_slot] = roster
    table = simulate_seasons(rosters, _board, _settings, n_sims=n_sims)
    if table.empty or my_slot not in table.index:
        return None
    row = table.loc[my_slot]
    return {
        'wins': float(row['Wins']),
        'wins_p25': float(row['Wins P25']),
        'wins_p75': float(row['Wins P75']),
        'playoff_pct': float(row['Playoff %']),
        'points': float(row['Points']),
        'field_wins': float(table.drop(index=my_slot)['Wins'].mean()),
        'rank': int(table['Wins'].rank(ascending=False, method='min').loc[my_slot]),
        'teams': int(len(table)),
    }
