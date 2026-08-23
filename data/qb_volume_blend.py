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
  At K=200 dropbacks, Willis's 57-dropback sample carries about 22% weight,
  landing his blended rush share at roughly 3x Miami's own 2025 rate (0.061)
  without simply inheriting a backup's small-sample extremes.

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
# measured rush share and the league mean. Measured against
# stats_player_week_2025.csv: league mean rush share of dropbacks is 0.117
# (std 0.068), and a starter's full season runs ~550-650 dropbacks, so K=200
# lets a half-season of starts already carry most of the weight while a
# handful of relief appearances cannot. Re-verify with
# scripts/check_volume_conservation.py / scripts/compare_model_vs_market.py
# if this ever needs retuning.
QB_STYLE_BLEND_K = 200.0

# A QB needs at least this many career dropback-equivalents on record before
# his own rush share counts toward the LEAGUE mean used to shrink everyone
# else's small samples - one bad garbage-time game should not move the
# league reference point.
QB_LEAGUE_QUALIFY_DROPBACKS = 30.0

# Attempts/carries of evidence for a 50/50 weight between a QB's own
# per-unit efficiency (yards/attempt, TD rate, INT rate, completion rate,
# yards/carry) and the league mean. Deliberately smaller than
# QB_STYLE_BLEND_K: efficiency rates stabilize faster than a shot-selection
# tendency like rush share, and this is a materially lighter shrink than the
# volume side - it exists to tame a 3-5 game outlier sample, not to erase a
# real half-season efficiency signal the way the volume fix must for a raw
# backup-appearance rate.
QB_EFFICIENCY_BLEND_K = 100.0

# Used only when a team has no prior-season passing history at all (an
# expansion scenario, never observed in practice) - an explicit, documented
# last resort rather than a silent zero.
FALLBACK_LEAGUE_RUSH_SHARE = 0.12
FALLBACK_TEAM_DROPBACKS = 34.0

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


def blend_qb1_volume(
        identity_keys: np.ndarray, team_keys: np.ndarray, qb1_mask: np.ndarray,
        prior_history: pd.DataFrame | None, k: float = QB_STYLE_BLEND_K,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Blended per-game rates for QB1-selected rows; NaN (not applicable) elsewhere.

    ``identity_keys``/``team_keys``/``qb1_mask`` are plain numpy arrays,
    positionally aligned to the caller's current-position frame (``cur`` in
    ``build_weekly_projections``) - pass ``.to_numpy()`` views, not pandas
    Series, so integer position indexing below is unambiguous.

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

    team_capacity = derive_team_dropback_capacity(frame, team_col='game_team')
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
            w_eff = float(row['passing_attempts'] / (row['passing_attempts'] + QB_EFFICIENCY_BLEND_K))
            for dep in QB_BLEND_DEPENDENT_STATS['passing_attempts']:
                personal_rate = float(row[dep] / row['passing_attempts'])
                blended_rate = w_eff * personal_rate + (1 - w_eff) * league_pass_rate[dep]
                rates[dep][i] = blended_rate * blended_pass_attempts
        if row['rushing_attempts'] > 0:
            w_eff_rush = float(row['rushing_attempts'] / (row['rushing_attempts'] + QB_EFFICIENCY_BLEND_K))
            for dep in QB_BLEND_DEPENDENT_STATS['rushing_attempts']:
                personal_rate = float(row[dep] / row['rushing_attempts'])
                blended_rate = w_eff_rush * personal_rate + (1 - w_eff_rush) * league_rush_rate[dep]
                rates[dep][i] = blended_rate * blended_rush_attempts

        audit['personal_dropbacks'][i] = dropbacks
        audit['personal_rush_share'][i] = personal_rush_share
        audit['league_rush_share'][i] = league_rush_share
        audit['evidence_weight'][i] = w_self
        audit['blended_rush_share'][i] = blended_share
        audit['team_dropback_capacity'][i] = team_dropbacks
        audit['team_capacity_source'][i] = capacity_source

    return rates, audit
