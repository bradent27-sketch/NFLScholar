"""Team-constrained pass-catcher target conservation (V2 experiment).

WHY THIS EXISTS. Every WR/TE/RB target rate in the normal weekly-projection
loop is drawn from a LEAGUE-WIDE position rate (``pos_rate_arr`` in
``build_weekly_projections``), scaled only by the player's own expected snap
share. Nothing in that path checks whether the sum of one team's pass
catchers matches what that team's own quarterback is actually projected to
throw. The 2026-08-22 V2 audit measured the result directly: team targets
summed to 1.19x-1.56x team pass attempts depending on the week (every
measured team, every measured week), and receiving yards ran as high as
1.6x passing yards. The excess was never in the well-known names - the
top 6 pass catchers on a team were already close to real per-player marks
(17.9 projected vs 18.5 real for ranks 1-3). It was in the tail: 13th+
option on a team's own board, individually modest, compounding across 415
such players leaguewide into 18% of all projected targets, against roughly
1% in real box scores.

This module is a post-assembly normalization pass, applied to the fully
concatenated board (all four positions already projected), the same seam
``redistribute_v2_vacated_usage`` already uses. It does not re-derive any
player's role; it takes the model's own already-computed target value as a
credibility score and fits the team's pass catchers to a real budget:

  1. The team's own projected quarterback pass attempts for the week (or, if
     that QB room has no live volume yet, a prior-season team-game target
     average - see ``derive_team_target_capacity``) times the league target-
     per-attempt ratio (targets are not quite 1:1 with attempts: throwaways,
     spikes, and batted balls at the line have no targeted receiver).
  2. The TRUSTED_TIER highest-projected pass catchers on the team keep their
     own value untouched - that is the range the audit found was already
     accurate, and there is no reason to refit a good number.
  3. Whatever capacity remains goes to the rest of the team's pool,
     proportional to each player's own current value, so a real committee
     role still outranks a deep bench name without either one being pinned
     to a fabricated share.
  4. Receptions, receiving yards, and receiving TDs are rescaled by the same
     per-player factor as targets, so each player's OWN catch rate and
     yards/TD-per-target are preserved exactly - only the target volume that
     efficiency multiplies against changes.

Every team gets a ledger row recording its capacity, source, and the
trusted/tail split, mirroring the audit trail already used by
``data.rb_role_allocator``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# The tier of a team's own pass catchers whose CURRENT projected value is
# trusted as-is rather than refit to the team budget. Set from the audit's
# own finding ("top 6 pass catchers per team are accurate"); widened by two
# to avoid trimming a real second-tier target earner on a genuinely
# receiver-heavy offense. Re-verify with scripts/check_volume_conservation.py
# if this ever needs retuning - it is a football judgment call, not a
# derived constant.
PASS_CAPACITY_TRUSTED_TIER = 8

# Used only when neither a live team pass-attempt total nor prior-season
# history is available for a team (never observed in practice - every real
# team throws the ball - kept as an explicit, documented last resort rather
# than a silent zero budget that would wipe out a team's receivers).
FALLBACK_TARGET_PER_ATTEMPT = 0.95

CAPACITY_LEDGER_COLUMNS = [
    'team', 'capacity', 'capacity_source', 'trusted_claim', 'tail_claim',
    'allocated', 'unallocated', 'trusted_count', 'tail_count', 'reason',
]


def derive_team_target_capacity(history: pd.DataFrame, team_col: str = 'team') -> pd.DataFrame:
    """Prior-season team-game target/attempt capacity, for teams with no live QB volume yet.

    Mirrors ``data.rb_role_allocator.derive_preseason_rb_capacities``: an
    average of TEAM-GAMES, not player-season sums, keyed on the immutable
    ``game_team`` field so a trade cannot misattribute an old game's targets
    to the player's new team. The league-wide target/attempt ratio (used to
    convert a LIVE projected pass-attempt total into a target budget) is
    attached as ``.attrs['target_per_attempt']`` on the returned frame rather
    than as a bare module constant, since it is measured from whichever
    season's data the caller supplies rather than hardcoded.
    """
    columns = ['team', 'team_target_capacity', 'team_pass_attempts_capacity',
               'capacity_games', 'capacity_source']
    empty = pd.DataFrame(columns=columns)
    empty.attrs['target_per_attempt'] = FALLBACK_TARGET_PER_ATTEMPT
    if history is None or history.empty or 'week' not in history.columns:
        return empty
    frame = history.copy()
    game_team_col = 'game_team' if 'game_team' in frame.columns else team_col
    if game_team_col not in frame.columns:
        return empty
    frame['_team'] = frame[game_team_col].astype(object).where(
        frame[game_team_col].notna(), '').astype(str).str.strip().str.upper()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[(frame['_team'] != '') & frame['_week'].notna()]
    if frame.empty:
        return empty
    for col in ('targets', 'passing_attempts'):
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
    team_week = frame.groupby(['_team', '_week'], observed=True).agg(
        _targets=('targets', 'sum'), _attempts=('passing_attempts', 'sum')
    ).reset_index()
    if team_week.empty or team_week['_attempts'].sum() <= 0:
        return empty
    capacity = team_week.groupby('_team', observed=True).agg(
        team_target_capacity=('_targets', 'mean'),
        team_pass_attempts_capacity=('_attempts', 'mean'),
        capacity_games=('_week', 'nunique'),
    ).reset_index().rename(columns={'_team': 'team'})
    capacity['capacity_source'] = 'prior-season team-game targets'
    result = capacity.reindex(columns=columns)
    result.attrs['target_per_attempt'] = float(
        team_week['_targets'].sum() / max(team_week['_attempts'].sum(), 1e-9))
    return result


def apply_pass_capacity_conservation(
        result: pd.DataFrame, prior_history: pd.DataFrame | None = None,
        team_col: str = 'team', tier_size: int = PASS_CAPACITY_TRUSTED_TIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one team's RB/WR/TE targets (and dependents) to a real pass budget.

    ``result`` is the fully assembled whole-league board (all four positions
    concatenated). Operates on ``Team``/``Pos``/``targets`` and, where
    present, ``receptions``/``receiving_yards``/``receiving_tds``. Returns
    ``(result, ledger)`` with the input's column set unchanged - only target
    and dependent-stat VALUES are rescaled, never dropped or added.
    """
    if result is None or result.empty or 'Team' not in result.columns or 'Pos' not in result.columns:
        return (result.copy() if result is not None else pd.DataFrame(),
               pd.DataFrame(columns=CAPACITY_LEDGER_COLUMNS))
    if 'targets' not in result.columns:
        return result.copy(), pd.DataFrame(columns=CAPACITY_LEDGER_COLUMNS)

    out = result.copy()
    out['targets'] = pd.to_numeric(out['targets'], errors='coerce').fillna(0.0)
    dependent_cols = [c for c in ('receptions', 'receiving_yards', 'receiving_tds') if c in out.columns]
    for col in dependent_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
    pass_attempts = pd.to_numeric(
        out['passing_attempts'], errors='coerce').fillna(0.0) if 'passing_attempts' in out.columns \
        else pd.Series(0.0, index=out.index)

    team_capacity = derive_team_target_capacity(prior_history, team_col=team_col)
    capacity_by_team = (team_capacity.set_index('team')['team_target_capacity'].to_dict()
                        if not team_capacity.empty else {})
    league_target_ratio = float(team_capacity.attrs.get('target_per_attempt', FALLBACK_TARGET_PER_ATTEMPT))

    teams = out['Team'].astype(str)
    catcher_mask = out['Pos'].isin(['RB', 'WR', 'TE'])
    qb_mask = out['Pos'].eq('QB')
    ledger: list[dict] = []

    for team, idx in out.index[catcher_mask].to_series().groupby(teams[catcher_mask]).groups.items():
        idx = pd.Index(idx)
        live_attempts = float(pass_attempts[qb_mask & teams.eq(team)].sum())
        if live_attempts > 0.5:
            capacity = live_attempts * league_target_ratio
            capacity_source = "live projected team pass attempts"
        else:
            fallback = capacity_by_team.get(str(team))
            capacity = float(fallback) if fallback is not None and np.isfinite(fallback) else np.nan
            capacity_source = "prior-season team-game targets (no live QB attempts)"
        if not np.isfinite(capacity) or capacity <= 0:
            ledger.append({'team': team, 'capacity': None, 'capacity_source': 'no capacity signal',
                          'trusted_claim': None, 'tail_claim': None, 'allocated': None, 'unallocated': None,
                          'trusted_count': 0, 'tail_count': 0,
                          'reason': 'No live QB attempts and no prior-season team history; left unmodified.'})
            continue

        current = out.loc[idx, 'targets']
        current_total = float(current.sum())
        if current_total <= 0:
            continue
        order = current.sort_values(ascending=False)
        trusted_idx = order.index[:tier_size]
        tail_idx = order.index[tier_size:]
        trusted_claim = float(current.loc[trusted_idx].sum())
        tail_claim = float(current.loc[tail_idx].sum())

        allocated = current.copy()
        if trusted_claim > capacity:
            # Even the trusted tier alone exceeds this team's realistic
            # budget (rare - e.g. an unresolved QB room forced the historical
            # fallback low). Scale the trusted tier down rather than pretend
            # the tail can still receive a share of an already-exhausted pool.
            factor = capacity / trusted_claim if trusted_claim > 0 else 0.0
            allocated.loc[:] = 0.0
            allocated.loc[trusted_idx] = current.loc[trusted_idx] * factor
            reason = 'Trusted tier alone exceeded capacity; scaled down, tail zeroed.'
        else:
            remaining = capacity - trusted_claim
            allocated.loc[trusted_idx] = current.loc[trusted_idx]
            if tail_claim > 0:
                allocated.loc[tail_idx] = current.loc[tail_idx] * (remaining / tail_claim)
            reason = 'Trusted tier retained; remaining tail scaled to the team budget.'

        factor_series = (allocated / current.replace(0.0, np.nan)).fillna(1.0)
        out.loc[idx, 'targets'] = allocated.round(3)
        for col in dependent_cols:
            out.loc[idx, col] = (out.loc[idx, col] * factor_series).round(3)

        ledger.append({
            'team': team, 'capacity': round(capacity, 2), 'capacity_source': capacity_source,
            'trusted_claim': round(trusted_claim, 2), 'tail_claim': round(tail_claim, 2),
            'allocated': round(float(allocated.sum()), 2),
            'unallocated': round(max(0.0, capacity - float(allocated.sum())), 2),
            'trusted_count': int(len(trusted_idx)), 'tail_count': int(len(tail_idx)), 'reason': reason,
        })

    return out, pd.DataFrame(ledger, columns=CAPACITY_LEDGER_COLUMNS)
