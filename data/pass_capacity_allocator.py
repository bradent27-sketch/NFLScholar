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
     to a fabricated share. The pass is SYMMETRIC (since 2026-08-29): a
     group whose whole projected claim falls UNDER its own budget is scaled
     UP by the same uniform budget/claim factor, rather than leaving the
     shortfall as silently lost team volume.
  4. Receptions, receiving yards, and receiving TDs are rescaled by the same
     per-player factor as targets, so each player's OWN catch rate and
     yards/TD-per-target are preserved exactly - only the target volume that
     efficiency multiplies against changes.

Every team gets a ledger row recording its capacity, source, and the
trusted/tail split, mirroring the audit trail already used by
``data.rb_role_allocator``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


# Retained only to label a "trusted" (top-N) vs "tail" count in the audit
# ledger - the fit itself is a single uniform proportional factor across the
# whole group (see _fit_group), so this number no longer changes any
# projection. Was the size of a protected tier until 2026-08-30; kept at 8
# so the ledger's trusted/tail split still reads sensibly.
PASS_CAPACITY_TRUSTED_TIER = 8

# Used only when neither a live team pass-attempt total nor prior-season
# history is available for a team (never observed in practice - every real
# team throws the ball - kept as an explicit, documented last resort rather
# than a silent zero budget that would wipe out a team's receivers).
FALLBACK_TARGET_PER_ATTEMPT = 0.95

# Deadband (in targets) around a position group's own budget within which
# NO adjustment is made at all - the claim is close enough to realistic that
# refitting would just add noise. Added 2026-08-29 per explicit request
# ("If it is within + or - 1 it should probably be left alone. If not, then
# probably can implement the fix."). Applies symmetrically to mild over- and
# under-claims; a group outside the band is fit exactly as before.
#
# TIGHTENED 1.0 -> 0.5 on 2026-09-07: the +-1 band was letting a re-shaped
# cold-start room stay ~1 target/player over budget uncorrected, and the
# cold-start signed-bias check (wk1-2 2022-25) had every position projecting
# HIGH - startable TE by +2.85 pts, 8 of 8 weeks. WEEKLY_CALIBRATION is being
# re-fit (per-season-phase) alongside this change.
PASS_CAPACITY_DEADBAND = 0.5

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

# WR-vs-TE SPLIT OF THE WR/TE RECONCILIATION (``wr_te_split=True``, wired
# on only for a cold start / weeks 1-2 - see apply_pass_capacity_conservation
# and the 'v2_wr_te_capacity_split' feature flag in data/weekly_projections.py).
#
# Without it, the WR/TE slice is fit with ONE uniform factor across every WR
# and TE alike. That is wrong at cold start specifically, when the reason a
# team's WR/TE claim is off its pass-attempt budget is almost always a
# WR-ROOM change the offseason hasn't fully re-primed in the model: a WR
# departs and his targets aren't reclaimed (claim << budget -> the symmetric
# pass scales the TE UP alongside the WRs - LAC's Gadsden after Keenan Allen,
# GB's Kraft after Doubs+Wicks, TB's Otton after Evans), or a high-volume WR
# is added (claim >> budget -> the TE is docked proportionally with the WRs -
# IND's Warren after Keenan Allen arrived). A WR-room change is ~80% a
# WR-volume event; the tight end should barely move.
#
# The fix mirrors the RB carve-out one level down, but splits the SIGNED
# OFF-BUDGET DELTA rather than the base claim: D = budget - (wr_claim +
# te_claim); the TE sub-budget is te_claim + w_te*D and the WR sub-budget
# absorbs the rest, then each sub-slice runs the identical uniform _fit_group
# it does today. Each tight end keeps his OWN prior-year target rate as the
# anchor (te_claim) - a genuine 12-personnel team's TE1 is untouched; only
# the reconciliation from a WR-room imbalance is damped for him.
#
# NOT done as a prior-season TE-share slice of the WR/TE pie (the exact RB
# design): that would clamp a legitimately ascendant tight end - a rookie
# stud, a new scheme - back down to the team's STALE prior-season TE share,
# and a TE role that just changed is precisely the case that breaks. The RB
# carve-out is safe from this because RB receiving share is far more stable
# year-to-year and a rookie back's receiving role does track team history.
#
# w_te = 0.20: a tight end's share of a MARGINAL team target is well below
# his share of total targets (leaguewide TE target share ~0.22-0.24 overall;
# the marginal/incremental target skews further to WRs). Env-overridable for
# a sweep, clipped to [0, 1].
def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(os.environ[name]), lo, hi))
    except (KeyError, TypeError, ValueError):
        return default


TE_MARGINAL_TARGET_WEIGHT = _env_float("TE_MARGINAL_TARGET_WEIGHT", 0.20, 0.0, 1.0)
# Ledger-label only (same role as PASS_CAPACITY_TRUSTED_TIER_RB): a team
# fields ~2 tight ends with any real target role, so the split TE sub-room's
# trusted/tail line reads sensibly. Does not change the fit math.
PASS_CAPACITY_TRUSTED_TIER_TE = 2

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
        wr_te_split: bool = False,
        te_marginal_target_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one team's RB/WR/TE targets (and dependents) to a real pass budget.

    ``result`` is the fully assembled whole-league board (all four positions
    concatenated). Operates on ``Team``/``Pos``/``targets`` and, where
    present, ``receptions``/``receiving_yards``/``receiving_tds``. Returns
    ``(result, ledger)`` with the input's column set unchanged - only target
    and dependent-stat VALUES are rescaled, never dropped or added.

    ``wr_te_split`` (default off; the weekly model turns it on only for a
    cold start / weeks 1-2 via the 'v2_wr_te_capacity_split' feature flag):
    reconcile the WR and TE sub-rooms against SEPARATE budgets instead of one
    uniform WR/TE factor, so a WR-room-driven off-budget delta (a departed or
    added wideout) lands ~80% on the WRs and only ``te_marginal_target_weight``
    of it on the tight ends. See TE_MARGINAL_TARGET_WEIGHT's module comment.
    Falls back to the single uniform fit when a team has no TE (or no WR) row,
    or when the room is already within the deadband of its combined budget.

    ``te_marginal_target_weight`` left at None reads the module-level
    ``TE_MARGINAL_TARGET_WEIGHT`` at call time, so monkeypatching that global
    (sweeps, env override) actually takes effect rather than being frozen to
    whatever it was when this def was evaluated.
    """
    if te_marginal_target_weight is None:
        te_marginal_target_weight = TE_MARGINAL_TARGET_WEIGHT
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

    player_detail: list[dict] = []

    def _fit_group(idx: pd.Index, group_capacity: float, group_tier: int,
                   detail_label: str = '') -> tuple[pd.Series, dict]:
        """Fit one position group's targets to its sub-budget with a SINGLE
        UNIFORM proportional factor - every player scaled by the same
        ``group_capacity / claim``, over budget or under, so a genuine WR1
        and a deep reserve take the same percentage move and neither is
        pinned to a fabricated share.

        History, all reverted 2026-08-30 at the user's request in favour of
        this uniform form: a trusted-tier-8 / tail split that kept the top 8
        whole when the budget allowed (2026-08-25); then a top-2 "anchor
        waterfall" that fully protected WR1/WR2/TE1 and made ranks 3+ absorb
        the entire over-budget amount (2026-08-29). The waterfall docked a
        real WR1 by zero while hammering WR4/WR5 ("not correct"), and the
        calibration refit showed it pushing raw top-of-board WR/TE
        projections up. Reducing a deep reserve's role is now entirely the
        job of the role model upstream (depth-chart caps, the buried-veteran
        dock), leaving this pass to do only a small, even, symmetric budget
        reconciliation. ``group_tier`` is kept only to label trusted/tail
        counts in the ledger; it no longer changes the math."""
        current = out.loc[idx, 'targets']
        order = current.sort_values(ascending=False)
        trusted_idx = order.index[:group_tier]
        tail_idx = order.index[group_tier:]
        trusted_claim = float(current.loc[trusted_idx].sum())
        tail_claim = float(current.loc[tail_idx].sum())
        current_total = float(current.sum())

        allocated = current.copy()
        if current_total <= 0:
            reason = 'No projected volume in this group; nothing to fit.'
        elif abs(current_total - group_capacity) <= PASS_CAPACITY_DEADBAND:
            # Close enough to a realistic budget that a refit would only add
            # noise - leave every player's own number exactly as projected.
            reason = (f"Claim {current_total:.1f} is within {PASS_CAPACITY_DEADBAND:.1f} of the "
                      f"{group_capacity:.1f} budget; left unadjusted.")
        else:
            # One uniform factor, applied to every player - symmetric for
            # over- and under-budget rooms alike.
            factor = group_capacity / current_total
            allocated.loc[:] = current.loc[:] * factor
            reason = (f"Claim {current_total:.1f} vs {group_capacity:.1f} budget; every player "
                      f"scaled {'up' if factor > 1.0 else 'down'} proportionally (x{factor:.3f}).")
        ledger_row = {
            'capacity': round(group_capacity, 2), 'trusted_claim': round(trusted_claim, 2),
            'tail_claim': round(tail_claim, 2), 'allocated': round(float(allocated.sum()), 2),
            'unallocated': round(max(0.0, group_capacity - float(allocated.sum())), 2),
            'trusted_count': int(len(trusted_idx)), 'tail_count': int(len(tail_idx)), 'reason': reason,
        }
        # Per-player room detail - who's actually in this team/group and
        # what the conservation pass did to each of them, not just the
        # team-level totals above. `team`/`group_label` are read from the
        # enclosing loop's current iteration (see call site) - safe because
        # _fit_group is always called and fully consumed within the same
        # iteration, never stored for later.
        tier_by_idx = {**{j: 'trusted' for j in trusted_idx}, **{j: 'tail' for j in tail_idx}}
        # The per-player room table in the UI groups on 'WR/TE'; a split
        # WR-only / TE-only fit still reports its rows under the whole
        # receiving room so that table keeps rendering (the split shows up
        # in the team-level ledger rows instead - see the call site).
        _dl = detail_label or group_label
        for j in idx:
            player_detail.append({
                'team': team, 'position_group': _dl,
                'player': out.loc[j, 'Player'] if 'Player' in out.columns else str(j),
                'position': out.loc[j, 'Pos'] if 'Pos' in out.columns else '',
                'tier': tier_by_idx.get(j, 'tail'),
                'targets_before': round(float(current.loc[j]), 3),
                'targets_after': round(float(allocated.loc[j]), 3),
            })
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

        # RB always fits as one slice. WR/TE fits as one uniform slice
        # (default), OR - when wr_te_split is on and the room has both a WR
        # and a TE and is materially off its combined budget - as two slices
        # whose budgets split the SIGNED off-budget delta by
        # te_marginal_target_weight. Anything else (TE-only room, WR-only
        # room, room already within the deadband) falls back to the single
        # slice unchanged. ``detail_label`` keeps the per-player room table
        # grouped under 'WR/TE' even for a split slice (see _fit_group).
        groups = [(rb_idx, rb_capacity, PASS_CAPACITY_TRUSTED_TIER_RB, 'RB', '')]
        wr_idx = other_idx[positions.loc[other_idx].eq('WR').to_numpy()]
        te_idx = other_idx[positions.loc[other_idx].eq('TE').to_numpy()]
        wr_claim = float(out.loc[wr_idx, 'targets'].sum())
        te_claim = float(out.loc[te_idx, 'targets'].sum())
        other_delta = other_capacity - (wr_claim + te_claim)
        if (wr_te_split and len(wr_idx) and len(te_idx)
                and abs(other_delta) > PASS_CAPACITY_DEADBAND):
            w_te = float(np.clip(te_marginal_target_weight, 0.0, 1.0))
            te_budget = max(0.0, te_claim + w_te * other_delta)
            wr_budget = max(0.0, other_capacity - te_budget)
            groups.append((wr_idx, wr_budget, tier_size, 'WR', 'WR/TE'))
            groups.append((te_idx, te_budget, PASS_CAPACITY_TRUSTED_TIER_TE, 'TE', 'WR/TE'))
        else:
            groups.append((other_idx, other_capacity, tier_size, 'WR/TE', ''))

        for group_idx, group_capacity, group_tier, group_label, detail_label in groups:
            if not len(group_idx) or float(out.loc[group_idx, 'targets'].sum()) <= 0:
                continue
            allocated, ledger_row = _fit_group(group_idx, group_capacity, group_tier, detail_label)
            factor_series = (allocated / out.loc[group_idx, 'targets'].replace(0.0, np.nan)).fillna(1.0)
            out.loc[group_idx, 'targets'] = allocated.round(3)
            for col in dependent_cols:
                out.loc[group_idx, col] = (out.loc[group_idx, col] * factor_series).round(3)
            ledger.append({'team': team, 'position_group': group_label,
                          'capacity_source': capacity_source, **ledger_row})

    ledger_df = pd.DataFrame(ledger, columns=CAPACITY_LEDGER_COLUMNS)
    # Attached via .attrs rather than a third return value so every existing
    # `out, ledger = apply_pass_capacity_conservation(...)` call site (the
    # real caller in weekly_projections.py and every test in
    # tests/test_pass_capacity_allocator.py) keeps working unchanged - same
    # pattern derive_team_target_capacity already uses for target_per_attempt.
    ledger_df.attrs['player_detail'] = pd.DataFrame(
        player_detail,
        columns=['team', 'position_group', 'player', 'position', 'tier',
                 'targets_before', 'targets_after'])
    return out, ledger_df
