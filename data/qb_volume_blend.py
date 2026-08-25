"""QB1 volume blend: team dropback capacity x evidence-weighted player style (V2 experiment).

WHY THIS EXISTS. The QB1 gate (``qb1_override``) correctly identifies which
one quarterback on a team should carry volume, then hands him his own raw
per-game rate from whatever games he happens to have played - full seasons
starting are read correctly, but a QB1 whose only prior-season evidence is a
handful of BACKUP relief appearances gets that backup-appearance rate at
full confidence, with the team's own passing volume never entering the
computation at all. Malik Willis, projected as Miami's 2026 Week 1 starter
off 4 relief appearances for Tennessee (7.74 attempts/game, 97.6 yards/game
- both far below a real starter's line, and both a backup-usage artifact,
not his true talent), is the case that exposed it.

The user's own diagnosis, verbatim: "Prior season rate should have some
influence on the rate this season...not just the team...team strategies
change with new players incorporated...particularly in a case like malik
willis a strong rushing qb in comparison to miami last year. Design a model
that blends the team and player tendencies."

This module is that blend, decomposed into three independent pieces:

  VOLUME is a TEAM property. How many total dropbacks (pass attempts + QB
  carries - sacks are not football production and are deliberately excluded,
  consistently, on both sides of every ratio below) a team runs per game is
  set by its offensive identity, not by which arm is throwing. Estimated from
  that team's own prior-season team-games (see ``derive_team_dropback_capacity``,
  the same "average of team-games" pattern as ``data.rb_role_allocator``).

  STYLE is a PLAYER property. How a QB splits his own dropbacks between
  passing and rushing is real, durable signal about him individually - Malik
  Willis actually does run more than a typical NFL QB - but a 4-game backup
  sample is thin evidence, so it is shrunk toward the league mean by how much
  evidence actually exists: ``w_self = personal_dropbacks / (personal_dropbacks + K)``.
  At K=10 dropbacks (lowered from 200, then 30, on 2026-08-25 - see
  QB_STYLE_BLEND_K), even a moderate half-season sample already carries
  ~96% self-weight, and Willis's 56-dropback backup-relief sample still
  gets a real, if much lighter, correction (~85% self-weight) rather than
  inheriting its small-sample extreme at full confidence - QB rushing is
  overwhelmingly an individual athleticism/willingness trait, not a team-
  scheme effect, so an established rusher's own measured share should
  barely be diluted at all.

  EFFICIENCY (yards/attempt, completion%, TD rate, INT rate, yards/carry) is
  a PLAYER property, but it needs the SAME evidence discipline as style: his
  own measured rate is the right center, shrunk toward a league per-unit
  rate by how much of his own evidence actually exists. Skipping this for a
  thin sample reintroduces the identical bug one level down - Malik Willis's
  4-game 2025 sample (35 attempts) carries a 12.06 yards/attempt raw rate
  (a real fifth of it is one 21-of-288 relief outing), far above any real
  starter's sustained rate, and naively multiplying that against a newly-
  correct volume would replace an under-projection with an equally wrong
  over-projection. Never borrowed from the TEAM (an efficiency stat is not a
  team property the way volume is) - only from the league-wide per-unit
  rate among qualified QBs, at a separate, tighter evidence threshold than
  the style blend (efficiency stabilizes faster than shot selection).

Returns per-game BLENDED RATES (not final projections - matchup, pace, and
availability still apply downstream exactly as they do for every other stat),
meant to replace a QB1's raw ``prior_rate`` before it enters
``_blended_rate``. That insertion point matters: as a new starter's own
current-season sample grows, ``_blended_rate``'s existing evidence weighting
naturally shifts weight off this blended prior and onto his real observed
starts, so the correction fades out on its own rather than needing a
separate in-season cutover rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.player_aliases import stable_roster_identity_keys


# Dropback-equivalents of evidence for a 50/50 weight between a QB's own
# measured rush share and the league mean. First lowered from 200 to 30 on
# 2026-08-25, then to 10 the same day after the user reviewed the board and
# said it still wasn't trusting a QB's own raw rushing numbers enough: QB
# rushing is almost entirely an athleticism/willingness trait of the
# individual player, not a team-scheme effect, so it should barely be
# blended toward the league mean at all once there is a real sample. At
# K=10, even a moderate half-season sample (~250 dropbacks) already reaches
# w_self~=0.96, and a full season (~600) is ~0.98 - essentially his own
# number. A genuine thin backup-relief sample (the Malik Willis case this
# constant was originally built for, ~56 dropbacks) still gets a real but
# now much lighter correction (w_self~=0.85) - this constant is no longer
# the primary defense against a small-sample outlier reading as gospel; it
# only softens the most extreme cases. Re-verify with
# scripts/check_volume_conservation.py / scripts/compare_model_vs_market.py
# if this ever needs retuning.
QB_STYLE_BLEND_K = 10.0

# A QB needs at least this many career dropback-equivalents on record before
# his own rush share counts toward the LEAGUE mean used to shrink everyone
# else's small samples - one bad garbage-time game should not move the
# league reference point.
QB_LEAGUE_QUALIFY_DROPBACKS = 30.0

# Attempts of evidence for a 50/50 weight between a QB's own per-unit
# PASSING efficiency (yards/attempt, TD rate, INT rate, completion rate) and
# the league mean. It exists to tame a 3-5 game outlier sample, not to erase
# a real half-season efficiency signal the way the volume fix must for a raw
# backup-appearance rate.
QB_PASS_EFFICIENCY_BLEND_K = 100.0

# Carries of evidence for the same 50/50 weight, but for RUSHING efficiency
# (yards/carry, rush TD rate) specifically. Split out from the passing
# constant on 2026-08-25, then lowered further the same day (20 -> 5) after
# the user reviewed the board and said the model still wasn't trusting a
# QB's own raw rushing numbers enough. A QB's per-carry rushing average is
# almost entirely his own athleticism/running style, not something that
# should regress toward a league mean off any real sample. At K=5, Kyler
# Murray's real 5.97 YPC on a thin 29-carry 2025 sample lands at ~5.66 (a
# ~5% pull, down from a 27% pull at the original shared K=100) - a real
# rushing QB with even a modest carry total now reads at close to his own
# number. A truly token sample (Joe Burrow's real 14 non-rushing-QB
# carries) still shrinks meaningfully (~74% self-weight, not full trust in
# 14 carries) since the evidence-weight formula is still in effect - it is
# just a much lighter touch than before.
QB_RUSH_EFFICIENCY_BLEND_K = 5.0

# Used only when a team has no prior-season passing history at all (an
# expansion scenario, never observed in practice) - an explicit, documented
# last resort rather than a silent zero.
FALLBACK_LEAGUE_RUSH_SHARE = 0.12
FALLBACK_TEAM_DROPBACKS = 34.0

# Two-season (2024) personal-style/efficiency blend, added 2026-08-24. Mirrors
# weekly_projections.py's PRIOR2_BLEND_* constants (same values, kept as
# independent constants here so this module stays self-contained rather than
# reaching back into its caller) - see that module's comment for the full
# rationale. Applied to PERSONAL rush share and PERSONAL per-unit efficiency
# BEFORE league shrinkage below, not to team dropback capacity (a team
# property, not something a QB's own down year should distort). This is the
# fix for a QB1 whose OWN prior-season sample was itself thin or depressed -
# Lamar Jackson/Jayden Daniels, an injury-shortened 2025 off a full, strong
# 2024 - which QB_STYLE_BLEND_K/QB_PASS_EFFICIENCY_BLEND_K/
# QB_RUSH_EFFICIENCY_BLEND_K alone cannot fix: they only shrink a thin
# sample toward the LEAGUE mean, never toward what this specific player has
# actually shown before.
QB_PRIOR2_FULL_SEASON_GAMES = 8.0
QB_PRIOR2_BASE_WEIGHT = 0.20
QB_PRIOR2_MAX_WEIGHT = 0.55
# Stay bullish on an ascending player: a blend that would LOWER a component
# (his 2024 was worse - e.g. a breakout 2025 off a thin/backup 2024) is cut
# to roughly a third weight; one that RAISES it (Lamar/Daniels) keeps the
# full weight above.
QB_PRIOR2_DECREASE_DAMPENING = 0.35
# A 2024 row under this many games is a practice-squad/inactive-all-year
# entry, not a season - ignored entirely, same as no row at all.
QB_PRIOR2_MIN_GAMES = 1.0

QB_BLEND_VOLUME_STATS = ('passing_attempts', 'rushing_attempts')
QB_BLEND_DEPENDENT_STATS = {
    'passing_attempts': ('passing_yards', 'passing_tds', 'passing_completions', 'passing_interceptions'),
    'rushing_attempts': ('rushing_yards', 'rushing_tds'),
}
_ALL_BLEND_STATS = QB_BLEND_VOLUME_STATS + tuple(
    stat for deps in QB_BLEND_DEPENDENT_STATS.values() for stat in deps)


def derive_team_dropback_capacity(history: pd.DataFrame, team_col: str = 'team') -> pd.DataFrame:
    """Prior-season team-game dropback-equivalents (pass attempts + QB carries).

    Same team-game-average pattern as ``data.rb_role_allocator.
    derive_preseason_rb_capacities`` and ``data.pass_capacity_allocator.
    derive_team_target_capacity``: keyed on the immutable ``game_team`` field
    where available so a midseason trade cannot misattribute an old game.
    """
    columns = ['team', 'team_dropback_capacity', 'capacity_games', 'capacity_source']
    empty = pd.DataFrame(columns=columns)
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
    for col in ('passing_attempts', 'rushing_attempts'):
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
    frame['_dropback'] = frame['passing_attempts'] + frame['rushing_attempts']
    team_week = frame.groupby(['_team', '_week'], observed=True)['_dropback'].sum().reset_index()
    capacity = team_week.groupby('_team', observed=True).agg(
        team_dropback_capacity=('_dropback', 'mean'), capacity_games=('_week', 'nunique'),
    ).reset_index().rename(columns={'_team': 'team'})
    capacity['capacity_source'] = 'prior-season team-game dropbacks'
    return capacity.reindex(columns=columns)


def _prior2_personal(prior2_history: pd.DataFrame | None, value_cols: tuple) -> tuple[pd.DataFrame, pd.Series]:
    """2024 per-identity summed stats + game counts, or empty if unusable.

    Mirrors the ``frame``/``personal`` construction for ``prior_history``
    below, kept as its own helper since it is optional and independent of
    the 2025 league/team-capacity computations.
    """
    empty_personal = pd.DataFrame(columns=list(value_cols) + ['_dropback'])
    empty_games = pd.Series(dtype=float)
    if prior2_history is None or prior2_history.empty:
        return empty_personal, empty_games
    frame2 = prior2_history
    if 'position' in frame2.columns:
        frame2 = frame2[frame2['position'].astype(str).str.upper().eq('QB')]
    if frame2.empty or 'week' not in frame2.columns:
        return empty_personal, empty_games
    name_col2 = next((c for c in ('player_display_name', 'Player', 'player', 'name') if c in frame2.columns), None)
    if name_col2 is None:
        return empty_personal, empty_games
    frame2 = frame2.copy()
    frame2['_identity_key'] = stable_roster_identity_keys(frame2, name_col2).astype(str)
    for col in value_cols:
        if col not in frame2.columns:
            frame2[col] = 0.0
        frame2[col] = pd.to_numeric(frame2[col], errors='coerce').fillna(0.0)
    frame2['_dropback'] = frame2['passing_attempts'] + frame2['rushing_attempts']
    personal2 = frame2.groupby('_identity_key', observed=True)[list(value_cols) + ['_dropback']].sum()
    games2 = frame2.groupby('_identity_key', observed=True)['week'].nunique()
    return personal2, games2


def _prior2_component_weight(games_2025: float, games_2024: float, dropbacks_2024: float) -> float:
    """0 if no usable 2024 read, else the base (pre-asymmetry) blend weight.

    Same shape as weekly_projections.py's PRIOR2_BLEND base-weight curve:
    flat QB_PRIOR2_BASE_WEIGHT for a normal 2025 sample, rising toward
    QB_PRIOR2_MAX_WEIGHT as 2025 games played falls toward zero.
    """
    if dropbacks_2024 <= 0 or games_2024 < QB_PRIOR2_MIN_GAMES:
        return 0.0
    fraction_missing = float(np.clip(
        (QB_PRIOR2_FULL_SEASON_GAMES - games_2025) / QB_PRIOR2_FULL_SEASON_GAMES, 0.0, 1.0))
    return QB_PRIOR2_BASE_WEIGHT + (QB_PRIOR2_MAX_WEIGHT - QB_PRIOR2_BASE_WEIGHT) * fraction_missing


def _blend_component_toward_prior2(current: float, prior2_value: float, weight_base: float) -> float:
    """Blend one personal share/rate toward its 2024 value, asymmetrically.

    A blend that would LOWER ``current`` (2024 was worse) is cut to
    QB_PRIOR2_DECREASE_DAMPENING of ``weight_base``; one that RAISES it
    keeps the full base weight - see the constants' module docstring.
    """
    if weight_base <= 0:
        return current
    weight = weight_base * QB_PRIOR2_DECREASE_DAMPENING if prior2_value < current else weight_base
    return (1.0 - weight) * current + weight * prior2_value


def blend_qb1_volume(
        identity_keys: np.ndarray, team_keys: np.ndarray, qb1_mask: np.ndarray,
        prior_history: pd.DataFrame | None, k: float = QB_STYLE_BLEND_K,
        prior2_history: pd.DataFrame | None = None,
        prior_history_team: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Blended per-game rates for QB1-selected rows; NaN (not applicable) elsewhere.

    ``identity_keys``/``team_keys``/``qb1_mask`` are plain numpy arrays,
    positionally aligned to the caller's current-position frame (``cur`` in
    ``build_weekly_projections``) - pass ``.to_numpy()`` views, not pandas
    Series, so integer position indexing below is unambiguous.

    ``prior_history`` (and ``prior2_history``) must already be filtered to
    the caller's own player-history eligibility flag (the same
    ``player_prior``/``player_prior2`` frames every other stat's prior-season
    rate uses - see ``annotate_player_history_participation`` in
    weekly_projections.py) before reaching this function. This function only
    reads PERSONAL per-player sums from them (``personal``/``personal2``,
    the qualified league pool for ``league_rush_share``/``league_pass_rate``/
    ``league_rush_rate``) - a QB-split/relief partial game left in here
    silently drags his own personal per-attempt rate toward that thin,
    low-usage relief line (confirmed real on Jayden Daniels' 2025: 205.6
    YPG over his 5 full-role games vs. 180.3 including his 2 QB-split relief
    games - almost exactly the gap this filtering closes).

    ``prior_history_team`` (optional, defaults to ``prior_history`` when not
    given - kept for any caller/test that only has one frame to give) is
    used ONLY for team dropback capacity below, and should be the RAW,
    UNFILTERED team-game history instead: an injured starter and his relief
    replacement still together describe the team's real dropback total that
    game, the same "raw team-game history" principle
    ``annotate_player_history_participation`` documents for defense profiles.

    ``prior2_history`` (2024, optional) blends into the PERSONAL rush share
    and PERSONAL per-unit efficiency below - before league shrinkage, never
    into team dropback capacity - so a QB1 whose own 2025 sample was itself
    thin or depressed (Lamar Jackson/Jayden Daniels: an injury-shortened
    2025 off a full, strong 2024) is read as more than just "shrink harder
    toward the league mean." See QB_PRIOR2_* above for the weight curve.

    Returns ``(rates, audit)``: ``rates`` maps each of the 6 QB stats this
    blend touches to a per-game rate array; ``audit`` carries the same shape
    for the decomposition (personal/league rush share, evidence weight, team
    capacity and its source) for the projection trace.
    """
    n = len(identity_keys)
    rates = {stat: np.full(n, np.nan) for stat in _ALL_BLEND_STATS}
    audit = {
        'personal_dropbacks': np.full(n, np.nan), 'personal_rush_share': np.full(n, np.nan),
        'league_rush_share': np.full(n, np.nan), 'evidence_weight': np.full(n, np.nan),
        'blended_rush_share': np.full(n, np.nan), 'team_dropback_capacity': np.full(n, np.nan),
        'team_capacity_source': np.full(n, '', dtype=object),
        'prior2_weight': np.full(n, np.nan), 'personal_dropbacks_2024': np.full(n, np.nan),
    }
    qb1_mask = np.asarray(qb1_mask, dtype=bool)
    if not qb1_mask.any() or prior_history is None or prior_history.empty:
        return rates, audit
    frame = prior_history
    if 'position' in frame.columns:
        frame = frame[frame['position'].astype(str).str.upper().eq('QB')]
    if frame.empty or 'week' not in frame.columns:
        return rates, audit
    name_col = next((c for c in ('player_display_name', 'Player', 'player', 'name') if c in frame.columns), None)
    if name_col is None:
        return rates, audit
    frame = frame.copy()
    frame['_identity_key'] = stable_roster_identity_keys(frame, name_col).astype(str)
    value_cols = ('passing_attempts', 'rushing_attempts', 'passing_yards', 'passing_tds',
                 'passing_completions', 'passing_interceptions', 'rushing_yards', 'rushing_tds')
    for col in value_cols:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
    frame['_dropback'] = frame['passing_attempts'] + frame['rushing_attempts']

    personal = frame.groupby('_identity_key', observed=True)[list(value_cols) + ['_dropback']].sum()
    games = frame.groupby('_identity_key', observed=True)['week'].nunique()
    personal2, games2 = _prior2_personal(prior2_history, value_cols)
    qualified = personal[personal['_dropback'] >= QB_LEAGUE_QUALIFY_DROPBACKS]
    league_rush_share = (float((qualified['rushing_attempts'] / qualified['_dropback']).mean())
                         if not qualified.empty else FALLBACK_LEAGUE_RUSH_SHARE)
    # League per-unit efficiency, from the same qualified pool: passing
    # dependents per attempt, rushing dependents per carry. A qualified QB
    # with zero attempts (a pure runner) or zero carries cannot contribute a
    # divide-by-zero to either league mean.
    pass_qualified = qualified[qualified['passing_attempts'] > 0]
    league_pass_rate = {
        dep: (float((pass_qualified[dep] / pass_qualified['passing_attempts']).mean())
             if not pass_qualified.empty else 0.0)
        for dep in QB_BLEND_DEPENDENT_STATS['passing_attempts']
    }
    rush_qualified = qualified[qualified['rushing_attempts'] > 0]
    league_rush_rate = {
        dep: (float((rush_qualified[dep] / rush_qualified['rushing_attempts']).mean())
             if not rush_qualified.empty else 0.0)
        for dep in QB_BLEND_DEPENDENT_STATS['rushing_attempts']
    }

    # Team dropback capacity deliberately reads the RAW (unfiltered) history,
    # not the eligibility-filtered `frame` used for personal/league rates
    # above: a QB-split game's two rows (starter's partial line + reliever's
    # partial line) still sum to the team's one real dropback total that
    # game, same as any other team-game aggregate in this app.
    team_source = prior_history_team if prior_history_team is not None else prior_history
    if 'position' in team_source.columns:
        team_source = team_source[team_source['position'].astype(str).str.upper().eq('QB')]
    team_capacity = derive_team_dropback_capacity(team_source, team_col='game_team')
    capacity_by_team = (team_capacity.set_index('team')['team_dropback_capacity'].to_dict()
                        if not team_capacity.empty else {})
    league_dropback_mean = (float(team_capacity['team_dropback_capacity'].mean())
                            if not team_capacity.empty else FALLBACK_TEAM_DROPBACKS)

    for i in np.flatnonzero(qb1_mask):
        key = identity_keys[i]
        if key not in personal.index:
            continue
        row = personal.loc[key]
        dropbacks = float(row['_dropback'])
        if dropbacks <= 0:
            continue
        personal_rush_share = float(row['rushing_attempts'] / dropbacks)

        row2 = personal2.loc[key] if key in personal2.index else None
        dropbacks2 = float(row2['_dropback']) if row2 is not None else 0.0
        prior2_weight = _prior2_component_weight(
            float(games.get(key, 0.0)), float(games2.get(key, 0.0)), dropbacks2)
        if prior2_weight > 0:
            personal_rush_share_2024 = float(row2['rushing_attempts'] / dropbacks2)
            personal_rush_share = _blend_component_toward_prior2(
                personal_rush_share, personal_rush_share_2024, prior2_weight)

        w_self = dropbacks / (dropbacks + k)
        blended_share = float(np.clip(w_self * personal_rush_share + (1 - w_self) * league_rush_share, 0.0, 0.95))

        team = str(team_keys[i])
        team_dropbacks = capacity_by_team.get(team)
        if team_dropbacks is None or not np.isfinite(team_dropbacks):
            team_dropbacks, capacity_source = league_dropback_mean, 'league-average team dropbacks (no team history)'
        else:
            capacity_source = 'prior-season team-game dropbacks'

        blended_rush_attempts = team_dropbacks * blended_share
        blended_pass_attempts = team_dropbacks * (1.0 - blended_share)
        rates['passing_attempts'][i] = blended_pass_attempts
        rates['rushing_attempts'][i] = blended_rush_attempts
        if row['passing_attempts'] > 0:
            w_eff = float(row['passing_attempts'] / (row['passing_attempts'] + QB_PASS_EFFICIENCY_BLEND_K))
            for dep in QB_BLEND_DEPENDENT_STATS['passing_attempts']:
                personal_rate = float(row[dep] / row['passing_attempts'])
                if prior2_weight > 0 and row2 is not None and row2['passing_attempts'] > 0:
                    personal_rate_2024 = float(row2[dep] / row2['passing_attempts'])
                    personal_rate = _blend_component_toward_prior2(personal_rate, personal_rate_2024, prior2_weight)
                blended_rate = w_eff * personal_rate + (1 - w_eff) * league_pass_rate[dep]
                rates[dep][i] = blended_rate * blended_pass_attempts
        if row['rushing_attempts'] > 0:
            w_eff_rush = float(row['rushing_attempts'] / (row['rushing_attempts'] + QB_RUSH_EFFICIENCY_BLEND_K))
            for dep in QB_BLEND_DEPENDENT_STATS['rushing_attempts']:
                personal_rate = float(row[dep] / row['rushing_attempts'])
                if prior2_weight > 0 and row2 is not None and row2['rushing_attempts'] > 0:
                    personal_rate_2024 = float(row2[dep] / row2['rushing_attempts'])
                    personal_rate = _blend_component_toward_prior2(personal_rate, personal_rate_2024, prior2_weight)
                blended_rate = w_eff_rush * personal_rate + (1 - w_eff_rush) * league_rush_rate[dep]
                rates[dep][i] = blended_rate * blended_rush_attempts

        audit['personal_dropbacks'][i] = dropbacks
        audit['personal_rush_share'][i] = personal_rush_share
        audit['league_rush_share'][i] = league_rush_share
        audit['prior2_weight'][i] = prior2_weight
        audit['personal_dropbacks_2024'][i] = dropbacks2 if row2 is not None else np.nan
        audit['evidence_weight'][i] = w_self
        audit['blended_rush_share'][i] = blended_share
        audit['team_dropback_capacity'][i] = team_dropbacks
        audit['team_capacity_source'][i] = capacity_source

    return rates, audit
