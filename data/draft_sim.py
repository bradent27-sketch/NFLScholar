"""
Mock draft simulator: a full draft against opponents who behave like real
drafters, under your actual league settings.

WHY A SIMULATOR AND NOT JUST A BOARD: a board tells you who's valuable, but
not what the room will leave you. The questions that actually decide a
draft - "if I go WR-WR at the 3/4 turn, is the RB room dead by my third
pick?", "from slot 11, does waiting on QB work in superflex?" - can only be
answered by running the draft. Running it fifty times answers it properly,
because one mock is one sample of a noisy process and it's the DISTRIBUTION
of outcomes that tells you whether a plan is sound or you got lucky.

THE OPPONENT MODEL: AI teams draft from ADP with noise, constrained by
roster legality. That combination is what makes them behave like a real
room rather than a ranked list being read off in order:

  - ADP with noise reproduces positional runs. Real drafts don't tick down
    ADP one pick at a time - somebody reaches for a QB, two more follow, and
    suddenly the position's gone. Sampling from a window rather than always
    taking the top name generates exactly that clustering, because a small
    reach early compounds through everyone after it.
  - Roster legality stops the nonsense that pure-ADP bots produce: nobody
    drafts a third kicker, and nobody enters the last round needing two
    starters. Real drafters fill their lineup, and a simulator whose bots
    don't will hand you a fantasy version of the draft where studs fall to
    you in round 8.

When ADP is unavailable the bots fall back to the board's own value
ranking, which is a worse opponent model (it drafts a "correct" draft, not
a realistic one) - flagged to the caller so the UI can say so rather than
quietly presenting it as a real mock.
"""
import numpy as np
import pandas as pd

from data.draft_board import (
    DRAFTABLE_POSITIONS, FLEX_ELIGIBLE, SUPERFLEX_ELIGIBLE, snake_pick_numbers,
)

# Kickers and defenses go at the very end of real drafts regardless of what
# a value model thinks, because they're replaceable weekly off waivers.
# Bots that draft them on ADP alone take them several rounds too early.
LATE_ROUND_ONLY = ['K', 'DST']
LATE_ROUND_WINDOW = 2

# Bench depth a sane drafter carries beyond their starting slots, per
# position. Added to the starting requirement rather than used as a flat
# cap, so the limits move with league settings: one backup QB in a 1QB
# league, several in superflex where every team needs two starters and the
# position runs dry.
BENCH_DEPTH_ALLOWANCE = {'QB': 1, 'RB': 4, 'WR': 4, 'TE': 1, 'K': 0, 'DST': 0}


def _position_limits(settings):
    """
    How many of each position anyone will roster, derived from this league's
    own settings rather than fixed constants.

    Deriving it matters because a flat cap is wrong in both directions at
    once. 'QB: 3' is far too many for a 1QB league - real drafters take one
    and maybe a backup, and bots that hoard three quietly starve the
    position and make your own QB look scarcer than it is - while being too
    FEW for superflex, where every team starts two and carrying three is
    normal. Same shape of error at TE between standard and TE-premium.
    """
    roster = settings['roster']
    flex = int(roster.get('FLEX', 0))
    superflex = int(roster.get('SUPERFLEX', 0))
    limits = {}
    for pos in DRAFTABLE_POSITIONS:
        starters = int(roster.get(pos, 0))
        if pos in SUPERFLEX_ELIGIBLE:
            starters += superflex
        if pos in FLEX_ELIGIBLE:
            # One flex slot is a claim shared across three positions, so it
            # counts as a third of a slot each rather than a whole one for
            # all of them.
            starters += flex / len(FLEX_ELIGIBLE)
        limits[pos] = max(1, int(round(starters + BENCH_DEPTH_ALLOWANCE.get(pos, 1))))
    return limits


def _starter_requirements(settings):
    """Starting slots a team must eventually fill, as {pos: count} plus flex counts."""
    roster = settings['roster']
    req = {pos: int(roster.get(pos, 0)) for pos in DRAFTABLE_POSITIONS}
    return req, int(roster.get('FLEX', 0)), int(roster.get('SUPERFLEX', 0))


def _unfilled_starter_slots(counts, settings):
    """How many starting slots this roster still has to fill."""
    req, flex, superflex = _starter_requirements(settings)
    remaining = 0
    leftover = {}
    for pos in DRAFTABLE_POSITIONS:
        have = counts.get(pos, 0)
        need = max(0, req.get(pos, 0) - have)
        remaining += need
        leftover[pos] = max(0, have - req.get(pos, 0))
    spare_flex = sum(leftover.get(p, 0) for p in FLEX_ELIGIBLE)
    remaining += max(0, flex - spare_flex)
    spare_superflex = sum(leftover.get(p, 0) for p in SUPERFLEX_ELIGIBLE)
    remaining += max(0, superflex - max(0, spare_superflex - flex))
    return remaining


# Roster construction ladder, applied after the starting lineup is full:
# one backup at each real position, and everything after that goes to
# running back and receiver.
#
# This is how people actually build a bench, and it is not what a pure
# value model does on its own - left alone it will happily take a third
# tight end or a second kicker in round 12 because they graded out highest,
# producing a roster nobody would draft. Backups at QB and TE are worth one
# spot each (you start one, so the second is insurance); backups at K and
# DST are worth none at all, since a bad week is solved off waivers.
BACKUP_ALLOWANCE = {'QB': 1, 'RB': 1, 'WR': 1, 'TE': 1, 'K': 0, 'DST': 0}
DEPTH_PRIORITY_POSITIONS = ['RB', 'WR']


def _construction_stage_positions(counts, settings):
    """
    Which positions the roster ladder allows next, or None for no restriction.

    Returns None while the starting lineup still has holes - at that point
    the value model and the need multiplier are already pulling in the right
    direction and don't need a hard filter on top.
    """
    roster = settings['roster']
    req, flex, superflex = _starter_requirements(settings)
    if _unfilled_starter_slots(counts, settings) > 0:
        return None

    # Starters are set: take one backup at each position that warrants one.
    #
    # Counted against DEDICATED slots only. A single FLEX slot is shared
    # between RB, WR and TE - adding it to each of their starter counts
    # tells the ladder every one of them is a slot short, which is how a
    # roster ends up with four tight ends in a league that starts one.
    # Flex-driven depth is handled where it belongs, in the final stage,
    # which puts extra picks into running backs and receivers.
    wants_backup = []
    for pos in DRAFTABLE_POSITIONS:
        started = req.get(pos, 0)
        if pos in SUPERFLEX_ELIGIBLE:
            started += superflex
        if started <= 0:
            continue
        if counts.get(pos, 0) < started + BACKUP_ALLOWANCE.get(pos, 0):
            wants_backup.append(pos)
    if wants_backup:
        return wants_backup
    return list(DEPTH_PRIORITY_POSITIONS)


def _legal_positions(counts, settings, round_no, total_rounds, picks_left):
    """
    Which positions this team may still draft at this point.

    The 'must fill starters' clamp at the end is the piece that makes bots
    behave: once a team has exactly as many picks left as unfilled starting
    slots, it stops taking best-available and starts filling holes - which
    is what any real drafter does, and it's why studs at a position everyone
    already has DO fall in the late rounds while a scarce starter never does.
    """
    limits = _position_limits(settings)
    req, _, superflex = _starter_requirements(settings)
    legal = []
    for pos in DRAFTABLE_POSITIONS:
        if counts.get(pos, 0) >= limits.get(pos, 99):
            continue
        if pos in LATE_ROUND_ONLY:
            if round_no < total_rounds - LATE_ROUND_WINDOW + 1:
                continue
            if req.get(pos, 0) == 0:
                continue
        legal.append(pos)

    # No backups anywhere until the starting lineup is complete. Nobody
    # takes a second quarterback in round three of a one-QB league while
    # their WR2 slot is empty, however well he grades - the backup can only
    # ever contribute if the starter is hurt, so a starting slot is strictly
    # more valuable than any bench copy of a position you've already filled.
    if _unfilled_starter_slots(counts, settings) > 0:
        flex_open = int(settings['roster'].get('FLEX', 0)) + int(settings['roster'].get('SUPERFLEX', 0)) > 0
        still_needed = []
        for pos in legal:
            dedicated = req.get(pos, 0)
            if pos in SUPERFLEX_ELIGIBLE:
                dedicated += superflex
            if counts.get(pos, 0) < dedicated:
                still_needed.append(pos)
            elif flex_open and pos in FLEX_ELIGIBLE:
                still_needed.append(pos)
        if still_needed:
            legal = still_needed

    ladder = _construction_stage_positions(counts, settings)
    if ladder:
        narrowed = [p for p in legal if p in ladder]
        if narrowed:
            legal = narrowed

    unfilled = _unfilled_starter_slots(counts, settings)
    if picks_left <= unfilled:
        needed = []
        for pos in DRAFTABLE_POSITIONS:
            if counts.get(pos, 0) < req.get(pos, 0):
                needed.append(pos)
        if not needed:
            needed = [p for p in (SUPERFLEX_ELIGIBLE if superflex else FLEX_ELIGIBLE)]
        legal = [p for p in legal if p in needed] or legal
    return legal


def _bot_pick(available, counts, settings, round_no, total_rounds, picks_left, rng, reach_window, order_col):
    """
    One bot pick: sample from the top of the legal board rather than always
    taking the best name.

    The exponential weighting means the bot usually takes near the top but
    occasionally reaches - the frequency controlled by reach_window. That
    single knob is what separates a chalk-drafting room (everyone follows
    ADP exactly, and your board is worthless because nothing surprising
    happens) from the sharp/chaotic rooms people actually draft in.
    """
    legal = _legal_positions(counts, settings, round_no, total_rounds, picks_left)
    pool = available[available['Pos'].astype(str).str.upper().isin(legal)]
    if pool.empty:
        pool = available
    if pool.empty:
        return None
    pool = pool.nsmallest(min(len(pool), int(max(3, round(reach_window * 3)))), order_col)
    n = len(pool)
    weights = np.exp(-np.arange(n) / max(reach_window, 0.5))
    weights = weights / weights.sum()
    idx = rng.choice(n, p=weights)
    return pool.iloc[idx]


# How simulated opponents rank players, as weights over the three
# independent orderings the board carries. The default is a straight average
# of ADP and consensus rank, which is much closer to how a real room behaves
# than either alone: pure ADP drafts a room that has collectively memorised
# last week's market and never has an opinion, while pure ECR drafts a room
# of analysts who all read the same rankings and ignore what everyone else
# is doing. Real drafters do some of both.
#
# FFA starts at zero because it's optional - it only exists once an import
# has been made, and a weight that silently does nothing is worse than one
# you turn on deliberately.
BOT_RANK_WEIGHTS = {'ADP': 0.5, 'ECR': 0.5, 'FFA': 0.0}

# Which board column carries each component.
_BOT_RANK_COLUMNS = {'ADP': 'ADP', 'ECR': 'ECR', 'FFA': 'FFA Rank'}


def build_opponent_ranking(board, weights=None):
    """
    The single ordering simulated opponents draft from, blended out of ADP,
    consensus rank and an FFA import.

    Blending happens in RANK space, not on the raw numbers. ADP is a pick
    number, ECR is a rank and FFA Rank is a rank over a different (smaller)
    player set - averaging those directly would let whichever column happens
    to have the widest numeric spread dominate the result for reasons that
    have nothing to do with what it says. Ranking each component over this
    board first puts all three on one scale where a weight of 0.5 genuinely
    means half.

    Weights are renormalized PER PLAYER over whichever components actually
    have a value for him. That's what stops a partial source from acting as
    a penalty: an FFA export covering 260 players would otherwise push the
    other 440 down the board purely for being absent from it, which would
    hand you a mock where every unlisted player falls into your lap.

    Returns (order_series, meta) with meta naming the components that
    contributed and the effective weights after dropping empty ones.
    """
    weights = {**BOT_RANK_WEIGHTS, **(weights or {})}
    ranks, used = {}, {}
    for name, column in _BOT_RANK_COLUMNS.items():
        weight = float(weights.get(name, 0.0) or 0.0)
        if weight <= 0 or column not in board.columns:
            continue
        values = pd.to_numeric(board[column], errors='coerce')
        # A component nobody on this board has is not a component. 30 is the
        # same threshold prepare_sim_pool has always used for "is there
        # really a market here".
        if values.notna().sum() < 30:
            continue
        ranks[name] = values.rank(method='first')
        used[name] = weight

    if not ranks:
        return None, {'components': {}, 'fallback': True}

    total = sum(used.values())
    used = {k: v / total for k, v in used.items()}

    numerator = pd.Series(0.0, index=board.index)
    denominator = pd.Series(0.0, index=board.index)
    for name, rank in ranks.items():
        present = rank.notna()
        numerator[present] += rank[present] * used[name]
        denominator[present] += used[name]

    blended = numerator.where(denominator > 0) / denominator.replace(0, np.nan)
    return blended, {'components': used, 'fallback': False}


def prepare_sim_pool(board, weights=None):
    """
    The draftable pool plus the ordering bots use.

    Bots follow the blended market ranking above where one can be built,
    and fall back to this board's own value order where it can't. The
    distinction is returned so the UI can be honest: a mock run against
    value-ordered bots is a useful stress test but it is not a simulation of
    a real draft room, and presenting it as one would make every "he'll fall
    to you" conclusion untrustworthy.
    """
    if board is None or board.empty:
        return pd.DataFrame(), None, False

    pool = board.copy()
    blended, meta = build_opponent_ranking(pool, weights)
    has_market = blended is not None and not meta.get('fallback')
    if has_market:
        # Players no component priced still need an order or they'd never be
        # drafted. Slot them behind everyone who was priced, ordered by this
        # board's own value - the right guess for "the room hasn't priced
        # him".
        worst = float(blended.max())
        fallback = pool['Board Rank'].astype(float) + worst
        pool['_order'] = blended.fillna(fallback)
    else:
        pool['_order'] = pool['Board Rank'].astype(float)
    pool = pool.dropna(subset=['_order'])
    pool.attrs['opponent_ranking'] = meta
    return pool.sort_values('_order').reset_index(drop=True), '_order', has_market


def init_draft_state(settings, my_slot, rounds, seed=None):
    """A fresh draft: nobody picked, everybody empty, clock at pick 1."""
    n_teams = int(settings['num_teams'])
    return {
        'num_teams': n_teams,
        'rounds': int(rounds),
        'my_slot': int(my_slot),
        'draft_type': settings.get('draft_type', 'Snake'),
        'pick_no': 1,
        'picks': [],                                   # [{pick, round, team, Player, Pos, ...}]
        'rosters': {t: [] for t in range(1, n_teams + 1)},
        'seed': seed if seed is not None else int(np.random.randint(0, 1_000_000)),
        'complete': False,
    }


def team_on_clock(state):
    """Whose pick it is, honoring snake ordering."""
    n = state['num_teams']
    pick_idx = state['pick_no'] - 1
    rnd = pick_idx // n
    slot = pick_idx % n
    if state.get('draft_type', 'Snake') == 'Snake' and rnd % 2 == 1:
        return n - slot
    return slot + 1


def current_round(state):
    return (state['pick_no'] - 1) // state['num_teams'] + 1


def _counts_for(state, team):
    counts = {}
    for p in state['rosters'][team]:
        pos = str(p.get('Pos', '')).upper()
        counts[pos] = counts.get(pos, 0) + 1
    return counts


def record_pick(state, team, player_row):
    """Commit one pick and advance the clock."""
    entry = {
        'pick': state['pick_no'],
        'round': current_round(state),
        'team': team,
        'Player': player_row['Player'],
        'Pos': str(player_row['Pos']).upper(),
        'Team NFL': player_row.get('Team'),
        'Proj Pts': float(player_row['Proj Pts']) if pd.notna(player_row.get('Proj Pts')) else np.nan,
        'VORP': float(player_row['VORP']) if pd.notna(player_row.get('VORP')) else np.nan,
        'ADP': float(player_row['ADP']) if pd.notna(player_row.get('ADP')) else np.nan,
        'Bye': player_row.get('Bye'),
        'Tier': player_row.get('Tier'),
    }
    state['picks'].append(entry)
    state['rosters'][team].append(entry)
    state['pick_no'] += 1
    if state['pick_no'] > state['num_teams'] * state['rounds']:
        state['complete'] = True
    return entry


def run_until_user_pick(state, settings, pool, order_col, reach_window=3.0):
    """
    Advance the draft through every bot pick until it's the user's turn (or
    the draft ends). Returns the picks that just happened, so the UI can
    show "here's what went between your picks" - which is the part of a mock
    people actually study.
    """
    made = []
    rng = np.random.default_rng(state['seed'] + state['pick_no'])
    drafted = {p['Player'] for p in state['picks']}
    total_picks = state['num_teams'] * state['rounds']

    while not state['complete']:
        team = team_on_clock(state)
        if team == state['my_slot']:
            break
        available = pool[~pool['Player'].isin(drafted)]
        if available.empty:
            state['complete'] = True
            break
        picks_left = state['rounds'] - len(state['rosters'][team])
        choice = _bot_pick(available, _counts_for(state, team), settings,
                           current_round(state), state['rounds'], picks_left,
                           rng, reach_window, order_col)
        if choice is None:
            state['complete'] = True
            break
        made.append(record_pick(state, team, choice))
        drafted.add(choice['Player'])
        if state['pick_no'] > total_picks:
            state['complete'] = True
    return made


def available_players(state, pool):
    drafted = {p['Player'] for p in state['picks']}
    return pool[~pool['Player'].isin(drafted)]


def optimal_lineup_points(roster, settings):
    """
    Best legal starting lineup from a roster, and what it projects for.

    Greedy by position then flex, which is optimal here because flex
    eligibility is nested (anyone flex-eligible is also eligible for their
    own dedicated slot) - so filling dedicated slots first with the best
    available at each position can never strand a better flex option.
    """
    roster_cfg = settings['roster']
    by_pos = {}
    for p in roster:
        by_pos.setdefault(str(p.get('Pos', '')).upper(), []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda r: (r.get('Proj Pts') if pd.notna(r.get('Proj Pts')) else 0), reverse=True)

    used, total = [], 0.0
    for pos in DRAFTABLE_POSITIONS:
        for _ in range(int(roster_cfg.get(pos, 0))):
            if by_pos.get(pos):
                pick = by_pos[pos].pop(0)
                used.append(pick)
                total += pick.get('Proj Pts') or 0
    for slot_name, eligible in (('FLEX', FLEX_ELIGIBLE), ('SUPERFLEX', SUPERFLEX_ELIGIBLE)):
        for _ in range(int(roster_cfg.get(slot_name, 0))):
            best, best_pos = None, None
            for pos in eligible:
                if by_pos.get(pos) and (best is None or (by_pos[pos][0].get('Proj Pts') or 0) > (best.get('Proj Pts') or 0)):
                    best, best_pos = by_pos[pos][0], pos
            if best is not None:
                by_pos[best_pos].pop(0)
                used.append(best)
                total += best.get('Proj Pts') or 0
    return round(total, 1), used


def grade_draft(state, settings):
    """
    Every team's projected starting-lineup points, ranked.

    Graded on the optimal STARTING lineup rather than total roster points on
    purpose: a team that drafted five good running backs cannot start five
    running backs, and summing the whole roster would reward exactly the
    hoarding strategy a good draft tool should be talking you out of. Bench
    depth is shown separately rather than folded into the grade.
    """
    rows = []
    for team, roster in state['rosters'].items():
        starters_pts, starters = optimal_lineup_points(roster, settings)
        bench_pts = round(sum((p.get('Proj Pts') or 0) for p in roster) - starters_pts, 1)
        rows.append({
            'Team': f"Team {team}" + (" (you)" if team == state['my_slot'] else ""),
            'Slot': team,
            'Starters Proj': starters_pts,
            'Bench Proj': bench_pts,
            'Roster': ", ".join(f"{p['Player']} ({p['Pos']})" for p in roster),
        })
    df = pd.DataFrame(rows).sort_values('Starters Proj', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    return df


BENCH_VALUE_DISCOUNT = 0.35


def marginal_lineup_gain(roster, candidate, settings):
    """
    How many projected points a candidate adds to the best legal STARTING
    lineup, plus a discounted credit for pure bench value.

    This is the objective a drafter actually has, and it differs from raw
    value in exactly the place that matters. A backup QB might carry a large
    VORP and still add nothing at all to your lineup, because you can only
    start one - so drafting on VORP alone produces rosters with three
    quarterbacks and no third receiver. Measuring the marginal starting-
    lineup gain instead makes positional need fall out of the arithmetic
    rather than needing a hand-tuned need bonus bolted on top.

    The bench term stops the opposite failure: with a pure starters-only
    objective, every bench pick is worth exactly zero and the last eight
    rounds become arbitrary. Depth genuinely is worth something (byes,
    injuries, in-season upside), just clearly less than a starting slot -
    hence a discount rather than full credit.
    """
    base, _ = optimal_lineup_points(roster, settings)
    with_candidate, _ = optimal_lineup_points(roster + [candidate], settings)
    gain = with_candidate - base
    if gain > 0:
        return gain
    depth_value = candidate.get('VORP')
    if depth_value is None or pd.isna(depth_value):
        depth_value = 0.0
    return max(0.0, float(depth_value)) * BENCH_VALUE_DISCOUNT


def autopick_for_user(state, settings, pool, value_col='VORP', explore=0.0, rng=None):
    """
    Best available for THIS roster - scored by marginal starting-lineup gain
    (see marginal_lineup_gain), not by raw board value.

    Only the top candidates by raw value are re-scored, because evaluating a
    lineup per candidate over the whole board would be wasted work: nothing
    outside the top of the board is going to win on marginal value anyway.
    """
    available = available_players(state, pool)
    if available.empty:
        return None
    counts = _counts_for(state, state['my_slot'])
    my_roster = state['rosters'][state['my_slot']]
    picks_left = state['rounds'] - len(my_roster)
    legal = _legal_positions(counts, settings, current_round(state), state['rounds'], picks_left)
    pool_legal = available[available['Pos'].astype(str).str.upper().isin(legal)]
    if pool_legal.empty:
        pool_legal = available
    # Candidates come off BOARD RANK where it exists, not raw projected
    # points or bare VORP.
    #
    # This matters most in round one. Raw value hands an empty roster the
    # quarterback every time - he really does project ~150 points above the
    # best back, and a lineup-gain calculation on an empty lineup sees
    # exactly that. What it cannot see is that a quarterback worth almost as
    # much is available eight rounds later. Board Rank already carries that
    # correction (market blend, late-round demotion), and it is the ordering
    # measured against expert consensus, so the simulator should draft the
    # same board the app recommends rather than a second, more naive one.
    if 'Board Rank' in pool_legal.columns and pool_legal['Board Rank'].notna().any():
        shortlist = pool_legal.nsmallest(min(len(pool_legal), 40), 'Board Rank')
    else:
        col = value_col if value_col in pool_legal.columns and pool_legal[value_col].notna().any() else 'Proj Pts'
        shortlist = pool_legal.nlargest(min(len(pool_legal), 40), col)
    has_board_rank = 'Board Rank' in shortlist.columns and shortlist['Board Rank'].notna().any()
    scores, rows = [], []
    for _, row in shortlist.iterrows():
        candidate = {'Pos': str(row['Pos']).upper(),
                     'Proj Pts': row.get('Proj Pts'), 'VORP': row.get('VORP')}
        gain = marginal_lineup_gain(my_roster, candidate, settings)
        if has_board_rank and pd.notna(row.get('Board Rank')):
            # Board order leads; the lineup-gain term only separates players
            # sitting close together on it, which is where roster fit should
            # decide and where the board itself is least confident anyway.
            score = -float(row['Board Rank']) + 0.02 * gain
        else:
            score = gain
        scores.append(score)
        rows.append(row)
    if not rows:
        return None
    scores = np.asarray(scores, dtype=float)

    if explore <= 0:
        return rows[int(np.argmax(scores))]

    # Stochastic pick, for simulations that need a DISTRIBUTION of what you
    # might do rather than a single answer.
    #
    # A strict argmax makes every simulated draft take the same player at the
    # same pick, so "what position lands here" comes back 100% one position
    # every time - technically true of a robot, useless as advice. Real
    # drafters choose among several near-equivalent options, and it's that
    # variety that makes the resulting frequencies mean something. Softmax
    # over the marginal-value scores keeps the choice sensible while
    # admitting that a two-point edge is not a decision.
    rng = rng or np.random.default_rng()
    spread = float(np.std(scores)) or 1.0
    weights = np.exp((scores - scores.max()) / max(spread * float(explore), 1e-6))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return rows[int(np.argmax(scores))]
    return rows[int(rng.choice(len(rows), p=weights / total))]


def simulate_full_draft(board, settings, my_slot, rounds, seed=None, reach_window=3.0,
                        user_strategy='value'):
    """
    Run one complete draft start to finish with the user auto-picking.

    The building block for run_many_drafts below - on its own a single mock
    is barely evidence, since one seed's positional runs are as likely to
    flatter a plan as to expose it.
    """
    pool, order_col, has_adp = prepare_sim_pool(board, settings.get('bot_weights'))
    if pool.empty:
        return None
    state = init_draft_state(settings, my_slot, rounds, seed=seed)
    while not state['complete']:
        run_until_user_pick(state, settings, pool, order_col, reach_window=reach_window)
        if state['complete']:
            break
        choice = autopick_for_user(state, settings, pool,
                                   value_col='VORP' if user_strategy == 'value' else 'Proj Pts')
        if choice is None:
            state['complete'] = True
            break
        record_pick(state, state['my_slot'], choice)
    state['has_adp'] = has_adp
    return state


def run_many_drafts(board, settings, my_slot, rounds, n_sims=25, reach_window=3.0, seed0=0):
    """
    Run the same draft slot many times and summarize what actually happens.

    This is the output that answers real questions. "Ashton Jeanty was
    available at my pick in 3 of 25 drafts" is a usable fact; "he was there
    in my one mock" is not. Same for the outcome spread: a slot whose median
    starting lineup is 1,480 points but ranges 1,300-1,620 is a materially
    different draft position than one that reliably lands 1,450, and the
    only way to see that is to run it repeatedly.

    Returns (summary_df_of_your_picks, outcomes_df).
    """
    pick_rows, outcomes = [], []
    for i in range(int(n_sims)):
        state = simulate_full_draft(board, settings, my_slot, rounds,
                                    seed=seed0 + i * 7919, reach_window=reach_window)
        if state is None:
            break
        mine = state['rosters'][my_slot]
        starters_pts, _ = optimal_lineup_points(mine, settings)
        grades = grade_draft(state, settings)
        my_rank = int(grades.loc[grades['Slot'] == my_slot, 'Rank'].iloc[0]) if not grades.empty else np.nan
        outcomes.append({'sim': i + 1, 'Starters Proj': starters_pts, 'League Rank': my_rank})
        for p in mine:
            pick_rows.append({'sim': i + 1, 'round': p['round'], 'Player': p['Player'], 'Pos': p['Pos']})

    if not pick_rows:
        return pd.DataFrame(), pd.DataFrame()

    picks_df = pd.DataFrame(pick_rows)
    n = picks_df['sim'].nunique()
    summary = (picks_df.groupby(['round', 'Player', 'Pos'])
               .size().reset_index(name='times'))
    summary['% of drafts'] = (summary['times'] / n * 100).round(0)
    summary = summary.sort_values(['round', 'times'], ascending=[True, False]).reset_index(drop=True)
    return summary, pd.DataFrame(outcomes)


def pick_slot_comparison(board, settings, rounds, n_sims=12, reach_window=3.0):
    """
    Run every draft slot to see which ones this league's settings actually
    favor.

    Worth having because the answer is genuinely league-specific and often
    surprising: superflex, TE premium and league size all move where the
    strong slots are, and the received wisdom about "the turn" is calibrated
    to 12-team PPR. If you get to pick your slot, this is the number that
    should decide it.
    """
    rows = []
    for slot in range(1, int(settings['num_teams']) + 1):
        _, outcomes = run_many_drafts(board, settings, slot, rounds,
                                      n_sims=n_sims, reach_window=reach_window, seed0=slot * 104729)
        if outcomes.empty:
            continue
        rows.append({
            'Draft Slot': slot,
            'Median Starters Proj': round(outcomes['Starters Proj'].median(), 1),
            'Best': round(outcomes['Starters Proj'].max(), 1),
            'Worst': round(outcomes['Starters Proj'].min(), 1),
            'Avg League Rank': round(outcomes['League Rank'].mean(), 2),
        })
    return pd.DataFrame(rows)
