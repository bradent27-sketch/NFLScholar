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

# RUNNING BACKS GET THEIR OWN SUB-BUDGET, SEPARATE FROM WR/TE. Added
# 2026-08-24 per a real, reported defect: this module used to rank a team's
# RB/WR/TE pass catchers TOGETHER by current target value and fit ONE shared
# trusted/tail split to the WHOLE group. That means a WR/TE-only change - a
# new signing, an injury, a rookie promoted - shifts who's "trusted" and how
# big the tail's claim is, which changes the tail's rescale factor for
# EVERY tail player, RB included, even though a real NFL running back's
# receiving role is governed by his own role on his own team and has
# essentially nothing to do with which wideouts/tight ends are on the roster
# ("running backs are chronically getting their receiving work negatively
# affected... entirely occurring due to change in the team's receiving
# depth/players (WR/TE), which truly doesn't really affect running backs").
# The fix: split the team's overall capacity (derived exactly as before, from
# live QB attempts or prior-season history - NOT from this week's WR/TE
# board) into an RB slice and a WR/TE slice using the team's own PRIOR-SEASON
# RB share of catcher targets, then run the identical trusted/tail fit
# separately within each slice. A WR/TE roster change can still move the
# WR/TE trusted/tail split same as before ("a slight ding in some cases" is
# still possible if the RB group's OWN claim exceeds ITS OWN sub-budget) but
# can never again move an RB's number by itself.
FALLBACK_RB_CATCHER_SHARE = 0.14
# A team essentially never fields more than one or two backs with a real
# receiving role - unlike WR/TE's trusted tier of 8, RB's is deliberately
# small so a genuine committee (2 backs both catching passes) still keeps
# both untouched, while a 3rd/4th reserve's incidental target draws from the
# same real budget instead of a fabricated league-average share.
PASS_CAPACITY_TRUSTED_TIER_RB = 2

CAPACITY_LEDGER_COLUMNS = [
    'team', 'position_group', 'capacity', 'capacity_source', 'trusted_claim', 'tail_claim',
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


def derive_team_rb_catcher_share(history: pd.DataFrame, team_col: str = 'team') -> tuple[dict, float]:
    """Each team's own prior-season RB share of its RB+WR+TE target total,
    plus a league-wide average as the fallback for a team with no such
    history. See PASS_CAPACITY_TRUSTED_TIER_RB's comment above for why this
    exists: it sizes an RB sub-budget that is independent of this week's
    WR/TE board.

    Team-game averaged, keyed on the immutable ``game_team`` field, for the
    same reason ``derive_team_target_capacity`` already is - a mid-season
    trade must not misattribute a game's targets to the wrong team.
    """
    if history is None or history.empty or 'week' not in history.columns or 'position' not in history.columns:
        return {}, FALLBACK_RB_CATCHER_SHARE
    frame = history.copy()
    game_team_col = 'game_team' if 'game_team' in frame.columns else team_col
    if game_team_col not in frame.columns:
        return {}, FALLBACK_RB_CATCHER_SHARE
    frame['_team'] = frame[game_team_col].astype(object).where(
        frame[game_team_col].notna(), '').astype(str).str.strip().str.upper()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame['_pos'] = frame['position'].astype(str).str.upper()
    frame = frame[(frame['_team'] != '') & frame['_week'].notna() & frame['_pos'].isin(['RB', 'WR', 'TE'])]
    if frame.empty:
        return {}, FALLBACK_RB_CATCHER_SHARE
    frame['targets'] = pd.to_numeric(frame.get('targets', 0.0), errors='coerce').fillna(0.0)
    team_week = frame.groupby(['_team', '_week', '_pos'], observed=True)['targets'].sum().unstack('_pos', fill_value=0.0)
    for col in ('RB', 'WR', 'TE'):
        if col not in team_week.columns:
            team_week[col] = 0.0
    catcher_total = team_week[['RB', 'WR', 'TE']].sum(axis=1)
    valid = catcher_total > 0
    if not valid.any():
        return {}, FALLBACK_RB_CATCHER_SHARE
    game_share = team_week.loc[valid, 'RB'] / catcher_total.loc[valid]
    league_share = float(game_share.mean())
    if not np.isfinite(league_share) or league_share <= 0:
        league_share = FALLBACK_RB_CATCHER_SHARE
    team_share = game_share.groupby(level='_team').mean().to_dict()
    return team_share, league_share


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
    rb_share_by_team, league_rb_share = derive_team_rb_catcher_share(prior_history, team_col=team_col)

    def _fit_group(idx: pd.Index, group_capacity: float, group_tier: int) -> tuple[pd.Series, dict]:
        """The trusted/tail fit, unchanged math, parameterized to run once
        per position group instead of once per team."""
        current = out.loc[idx, 'targets']
        order = current.sort_values(ascending=False)
        trusted_idx = order.index[:group_tier]
        tail_idx = order.index[group_tier:]
        trusted_claim = float(current.loc[trusted_idx].sum())
        tail_claim = float(current.loc[tail_idx].sum())
        current_total = float(current.sum())

        allocated = current.copy()
        if trusted_claim > group_capacity:
            # Even the trusted tier alone exceeds this group's realistic
            # budget. Originally this zeroed everyone outside the trusted
            # tier outright, on the assumption the case was rare and
            # pathological (an unresolved QB room forcing a low historical-
            # fallback budget). A real-data audit (2026-08-24, user-reported:
            # Derrick Henry and Kenneth Walker's receiving lines reading as
            # fully zeroed) found it is NOT rare - a run-heavy team with an
            # established low-target-share workhorse RB routinely has its
            # "trusted" top slots filled by WR/TE names alone (a lead back's
            # raw target rate is naturally lower than a team's top
            # wideouts), so the tail is not always the fringe of the roster;
            # it can contain a real, established player whose receiving role
            # is modest but genuine. Zeroing him outright was never intended
            # - now every catcher (trusted and tail alike) is scaled down by
            # the same factor, so a real-but-modest role degrades
            # proportionally instead of vanishing.
            factor = group_capacity / current_total if current_total > 0 else 0.0
            allocated.loc[:] = current.loc[:] * factor
            reason = 'Total claim exceeded its own sub-budget; everyone in this group scaled down proportionally.'
        else:
            remaining = group_capacity - trusted_claim
            allocated.loc[trusted_idx] = current.loc[trusted_idx]
            if tail_claim > 0:
                allocated.loc[tail_idx] = current.loc[tail_idx] * (remaining / tail_claim)
            reason = 'Trusted tier retained; remaining tail scaled to this group\'s own sub-budget.'
        ledger_row = {
            'capacity': round(group_capacity, 2), 'trusted_claim': round(trusted_claim, 2),
            'tail_claim': round(tail_claim, 2), 'allocated': round(float(allocated.sum()), 2),
            'unallocated': round(max(0.0, group_capacity - float(allocated.sum())), 2),
            'trusted_count': int(len(trusted_idx)), 'tail_count': int(len(tail_idx)), 'reason': reason,
        }
        return allocated, ledger_row

    teams = out['Team'].astype(str)
    positions = out['Pos'].astype(str)
    catcher_mask = positions.isin(['RB', 'WR', 'TE'])
    qb_mask = positions.eq('QB')
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
            ledger.append({'team': team, 'position_group': 'ALL', 'capacity': None,
                          'capacity_source': 'no capacity signal',
                          'trusted_claim': None, 'tail_claim': None, 'allocated': None, 'unallocated': None,
                          'trusted_count': 0, 'tail_count': 0,
                          'reason': 'No live QB attempts and no prior-season team history; left unmodified.'})
            continue

        # RB gets an independent slice of this SAME team-wide capacity - sized
        # from prior-season history, never from this week's WR/TE board - so
        # a WR/TE roster change can shift the WR/TE trusted/tail split without
        # ever touching an RB's number. See PASS_CAPACITY_TRUSTED_TIER_RB's
        # module comment for the full rationale.
        team_pos = positions.loc[idx]
        rb_idx = idx[team_pos.eq('RB').to_numpy()]
        other_idx = idx[team_pos.isin(['WR', 'TE']).to_numpy()]
        # Only carve out an RB slice when the team actually HAS a rostered RB
        # catcher this week - reserving a share for a position group with
        # nobody in it would just shrink WR/TE's real budget for nothing.
        if len(rb_idx) and len(other_idx):
            rb_share = float(rb_share_by_team.get(str(team), league_rb_share))
            rb_capacity = capacity * rb_share
        elif len(rb_idx):
            rb_capacity = capacity
        else:
            rb_capacity = 0.0
        other_capacity = capacity - rb_capacity

        for group_idx, group_capacity, group_tier, group_label in (
                (rb_idx, rb_capacity, PASS_CAPACITY_TRUSTED_TIER_RB, 'RB'),
                (other_idx, other_capacity, tier_size, 'WR/TE'),
        ):
            if not len(group_idx) or float(out.loc[group_idx, 'targets'].sum()) <= 0:
                continue
            allocated, ledger_row = _fit_group(group_idx, group_capacity, group_tier)
            factor_series = (allocated / out.loc[group_idx, 'targets'].replace(0.0, np.nan)).fillna(1.0)
            out.loc[group_idx, 'targets'] = allocated.round(3)
            for col in dependent_cols:
                out.loc[group_idx, col] = (out.loc[group_idx, col] * factor_series).round(3)
            ledger.append({'team': team, 'position_group': group_label,
                          'capacity_source': capacity_source, **ledger_row})

    return out, pd.DataFrame(ledger, columns=CAPACITY_LEDGER_COLUMNS)
