"""Team RB carry budget ('v2_rb_carry_budget', CANDIDATE 2026-09-30).

WHY. Every RB's carries are projected on his own (rate x role x multipliers)
and nothing checks the room's SUM against what that team actually hands its
backs. Measured on 2024-2025 boards (408 team-weeks, after the level fix and
v2_rb_participation): the board's RB room sum averaged 22.98 against 21.87
actual RB carries, with RMSE 7.62 and correlation 0.135 - noisier than a plain
team-level estimate (RMSE 6.1, corr 0.28). Per-player errors that are each
defensible add up to a room total that is both too high and too dispersed.

THE BUDGET (per team, per week, in-season only), fit by
scripts/fit_rb_carry_budget.py and stored in data/rb_carry_budget.json:

    budget = lg_rb + c0 + c_rate * (rate - lg_rate)
                        + c_spread * clip(spread, -SPREAD_CLIP, +SPREAD_CLIP)
                        + c_opp * (opp_rb_allowed - lg_rb)

    rate            the team's RB carries / offensive play this season to date,
                    shrunk toward last season with K_SHRINK games of weight
                    (plays = pass attempts + rush attempts, all positions)
    spread          closing spread from the team's side (+ = favored)
    opp_rb_allowed  RB carries per game the opponent's defense has allowed,
                    same shrinkage
    lg_*            last season's league means

What was tested and dropped (scripts/diag_rb_carry_budget.py, fit 2019-2022,
out of sample 2023-2025): team plays/game carries no RB-carry signal once the
RB rate is in (coefficient ~0); opponent pace adds nothing once opponent RB
carries allowed is in; the game total adds nothing. Spread matters mostly
through team quality, which rate and opp_rb_allowed already carry; what is
left is small and flat beyond a touchdown (hence the clip). Raw RB carries by
spread: underdogs of 3.5+ run ~1 carry below their own rate, favorites of 3.5+
~1.1 above, games inside a field goal ~0.

THE FIT. A room's claim counts a SIDELINED back at his pre-injury volume
(``_full_rushing_attempts``, the same injury-neutral claim pass capacity uses)
so vacancy stays the only pass that hands his carries out. When the claim is
within +-DEADBAND carries of the budget nothing moves (per the user: a 1-carry
buffer where the backfield is left alone). Outside it the room is pulled to the
NEAREST EDGE of the band (budget +- DEADBAND), not to the budget itself, so a
room 1.1 carries over moves 0.1, not 1.1 - there is no jump at the band edge.
Symmetric: an under-budget room is scaled up the same way (a down-only cap was
measurably worse, RMSE 6.64 vs 6.15, because the board under-claims too).

Distribution: uniform factor (every back the same %, the pass-capacity rule), or
with ``lead_weighted`` ('v2_rb_carry_budget_lead') the change is split in
proportion to carries^2, so the lead back absorbs most of it - the remaining
room excess sits almost entirely on RB1 (+1.29 of +1.64 carries per room).

Rushing yards and TDs move with carries (each back keeps his own yards/carry
and TD/carry). A sidelined back's stashed ``_full_rushing_*`` volume is scaled
by the same factor so vacancy redistributes budget-consistent carries.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from data.pass_capacity_allocator import SIDELINED_AVAILABILITY

RB_CARRY_BUDGET_PATH = os.path.join('data', 'rb_carry_budget.json')
K_SHRINK = 4.0
SPREAD_CLIP = 7.0
FALLBACK_LEAGUE_PLAYS = 62.0
FALLBACK_LEAGUE_RB_CARRIES = 21.9


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(os.environ[name]), lo, hi))
    except (KeyError, TypeError, ValueError):
        return default


RB_CARRY_BUDGET_DEADBAND = _env_float('RB_CARRY_BUDGET_DEADBAND', 1.0, 0.0, 10.0)
RUSH_DEPENDENTS = ('rushing_yards', 'rushing_tds')
LEDGER_COLUMNS = ['team', 'budget', 'claim', 'target', 'factor', 'rate', 'lg_rate', 'spread', 'opponent',
                  'opp_rb_allowed', 'lg_rb', 'team_games', 'reason']


@lru_cache(maxsize=2)
def load_rb_carry_budget_params(path: str = RB_CARRY_BUDGET_PATH):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def team_game_frame(stats: pd.DataFrame, team_col: str = 'team') -> pd.DataFrame:
    """One row per (team, week) from a player-week stat frame: offensive plays
    (pass attempts + rush attempts, every position), RB carries, opponent.
    Rows with no game (roster-only rows: no game_team) are ignored."""
    cols = ['team', 'week', 'opponent', 'plays', 'rb_carries']
    if stats is None or stats.empty or 'week' not in stats.columns:
        return pd.DataFrame(columns=cols)
    s = stats
    team = s['game_team'] if 'game_team' in s.columns else s.get(team_col)
    if team is None:
        return pd.DataFrame(columns=cols)
    opp = None
    for c in ('game_opponent', 'opponent_team'):
        if c in s.columns:
            opp = s[c] if opp is None else opp.fillna(s[c])
    frame = pd.DataFrame({
        'team': team.astype('string').str.upper(),
        'week': pd.to_numeric(s['week'], errors='coerce'),
        'opponent': (opp.astype('string').str.upper() if opp is not None else pd.Series(pd.NA, index=s.index, dtype='string')),
        'pass': pd.to_numeric(s.get('passing_attempts', 0.0), errors='coerce'),
        'rush': pd.to_numeric(s.get('rushing_attempts', 0.0), errors='coerce'),
        'pos': s['position'].astype(str) if 'position' in s.columns else '',
    }, index=s.index)
    frame = frame[frame['team'].notna() & frame['week'].notna()]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    frame[['pass', 'rush']] = frame[['pass', 'rush']].fillna(0.0)
    frame['rb'] = np.where(frame['pos'].eq('RB'), frame['rush'], 0.0)
    g = frame.groupby(['team', 'week'], observed=True).agg(
        opponent=('opponent', 'first'), passes=('pass', 'sum'), rushes=('rush', 'sum'), rb_carries=('rb', 'sum'))
    g['plays'] = g['passes'] + g['rushes']
    g = g[g['plays'] > 0].reset_index()
    return g[cols]


def _spread_by_team(schedule_df: pd.DataFrame, week: int) -> dict:
    if schedule_df is None or schedule_df.empty or not {'week', 'home_team', 'away_team'}.issubset(schedule_df.columns):
        return {}
    wk = schedule_df[pd.to_numeric(schedule_df['week'], errors='coerce') == week]
    spread = (pd.to_numeric(wk['spread_line'], errors='coerce') if 'spread_line' in wk.columns
              else pd.Series(np.nan, index=wk.index))
    out = {}
    for home, away, sp in zip(wk['home_team'].astype(str).str.upper(), wk['away_team'].astype(str).str.upper(), spread):
        out[home] = {'spread': sp, 'opponent': away}
        out[away] = {'spread': -sp if pd.notna(sp) else np.nan, 'opponent': home}
    return out


def budget_features(cur_games: pd.DataFrame, prior_games: pd.DataFrame, matchups: dict,
                    k: float = K_SHRINK) -> pd.DataFrame:
    """Per-team budget inputs for one target week. ``cur_games`` holds only
    games BEFORE the target week; ``matchups`` maps team -> {'spread', 'opponent'}."""
    prior_games = prior_games if prior_games is not None else pd.DataFrame(columns=cur_games.columns)
    if len(prior_games):
        lg_plays = float(prior_games['plays'].mean())
        lg_rb = float(prior_games['rb_carries'].mean())
    elif len(cur_games):
        lg_plays, lg_rb = float(cur_games['plays'].mean()), float(cur_games['rb_carries'].mean())
    else:
        lg_plays, lg_rb = FALLBACK_LEAGUE_PLAYS, FALLBACK_LEAGUE_RB_CARRIES
    lg_rate = lg_rb / lg_plays if lg_plays > 0 else FALLBACK_LEAGUE_RB_CARRIES / FALLBACK_LEAGUE_PLAYS
    p_off = prior_games.groupby('team').agg(p_plays=('plays', 'mean'), p_rb=('rb_carries', 'mean')) \
        if len(prior_games) else pd.DataFrame(columns=['p_plays', 'p_rb'])
    p_def = prior_games.groupby('opponent').agg(p_drb=('rb_carries', 'mean')) \
        if len(prior_games) else pd.DataFrame(columns=['p_drb'])
    c_off = cur_games.groupby('team').agg(n=('week', 'nunique'), plays=('plays', 'sum'), rb=('rb_carries', 'sum')) \
        if len(cur_games) else pd.DataFrame(columns=['n', 'plays', 'rb'])
    c_def = cur_games.groupby('opponent').agg(n=('week', 'nunique'), rb=('rb_carries', 'sum')) \
        if len(cur_games) else pd.DataFrame(columns=['n', 'rb'])
    rows = []
    for team, m in matchups.items():
        po_plays = float(p_off['p_plays'].get(team, lg_plays)) if len(p_off) else lg_plays
        po_rb = float(p_off['p_rb'].get(team, lg_rb)) if len(p_off) else lg_rb
        n = float(c_off['n'].get(team, 0.0)) if len(c_off) else 0.0
        plays = float(c_off['plays'].get(team, 0.0)) if len(c_off) else 0.0
        rb = float(c_off['rb'].get(team, 0.0)) if len(c_off) else 0.0
        rate = (rb + k * po_rb) / (plays + k * po_plays) if (plays + k * po_plays) > 0 else lg_rate
        opp = m.get('opponent')
        pd_rb = float(p_def['p_drb'].get(opp, lg_rb)) if len(p_def) else lg_rb
        nd = float(c_def['n'].get(opp, 0.0)) if len(c_def) else 0.0
        drb = float(c_def['rb'].get(opp, 0.0)) if len(c_def) else 0.0
        opp_rb = (drb + k * pd_rb) / (nd + k)
        spread = m.get('spread')
        rows.append({'team': team, 'opponent': opp, 'team_games': n, 'rate': rate, 'lg_rate': lg_rate,
                     'spread': float(spread) if spread is not None and pd.notna(spread) else 0.0,
                     'opp_rb_allowed': opp_rb, 'lg_rb': lg_rb})
    return pd.DataFrame(rows).set_index('team') if rows else pd.DataFrame()


def budget_design(features: pd.DataFrame) -> np.ndarray:
    """[1, rate - lg_rate, clip(spread), opp_rb_allowed - lg_rb] - the fit's
    and the model's single definition of the regressors."""
    return np.column_stack([
        np.ones(len(features)),
        (features['rate'] - features['lg_rate']).to_numpy(float),
        features['spread'].clip(-SPREAD_CLIP, SPREAD_CLIP).to_numpy(float),
        (features['opp_rb_allowed'] - features['lg_rb']).to_numpy(float),
    ])


def team_rb_carry_budgets(hist: pd.DataFrame, prior_stats: pd.DataFrame, schedule_df: pd.DataFrame,
                          week: int, team_col: str = 'team', prior_team_col: str = 'team',
                          params: dict | None = None) -> pd.DataFrame:
    """Budget per team for ``week``. ``hist`` must already be restricted to
    games before the as-of week. Empty frame when params are missing."""
    params = params if params is not None else load_rb_carry_budget_params()
    if not params:
        return pd.DataFrame()
    matchups = _spread_by_team(schedule_df, week)
    if not matchups:
        return pd.DataFrame()
    feats = budget_features(team_game_frame(hist, team_col), team_game_frame(prior_stats, prior_team_col),
                            matchups, k=float(params.get('k_shrink', K_SHRINK)))
    if feats.empty:
        return feats
    coef = np.array([params['intercept'], params['coefs']['rate'], params['coefs']['spread'],
                     params['coefs']['opp_rb']], dtype=float)
    feats['budget'] = feats['lg_rb'] + budget_design(feats) @ coef
    return feats


def apply_rb_carry_budget(result: pd.DataFrame, budgets: pd.DataFrame, deadband: float | None = None,
                          lead_weighted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each team's RB carries to its budget band. Returns (board, ledger)."""
    deadband = RB_CARRY_BUDGET_DEADBAND if deadband is None else float(deadband)
    if (result is None or result.empty or budgets is None or budgets.empty
            or 'rushing_attempts' not in result.columns or not {'Team', 'Pos'}.issubset(result.columns)):
        return (result.copy() if result is not None else pd.DataFrame()), pd.DataFrame(columns=LEDGER_COLUMNS)
    out = result.copy()
    carries = pd.to_numeric(out['rushing_attempts'], errors='coerce').fillna(0.0)
    avail = (pd.to_numeric(out['Availability'], errors='coerce').fillna(1.0) if 'Availability' in out.columns
             else pd.Series(1.0, index=out.index))
    sidelined = avail <= SIDELINED_AVAILABILITY
    full = carries.copy()
    if '_full_rushing_attempts' in out.columns:
        full = full.where(~sidelined, pd.to_numeric(out['_full_rushing_attempts'], errors='coerce').fillna(carries))
    rb = out['Pos'].astype(str).eq('RB')
    teams = out['Team'].astype(str).str.upper()
    ledger = []
    for team, idx in out.index[rb].to_series().groupby(teams[rb]).groups.items():
        idx = pd.Index(idx)
        info = {'team': team}
        if team in budgets.index:
            b = budgets.loc[team]
            for c in ('rate', 'lg_rate', 'spread', 'opponent', 'opp_rb_allowed', 'lg_rb', 'team_games'):
                v = b.get(c)
                info[c] = round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
        claim = float(full.loc[idx].sum())
        if team not in budgets.index or not np.isfinite(budgets.loc[team, 'budget']):
            ledger.append({**info, 'claim': round(claim, 2), 'reason': 'No budget for this team; left unmodified.'})
            continue
        budget = float(budgets.loc[team, 'budget'])
        row = {**info, 'budget': round(budget, 2), 'claim': round(claim, 2)}
        gap = claim - budget
        if claim <= 0:
            ledger.append({**row, 'target': round(claim, 2), 'factor': 1.0, 'reason': 'No projected RB carries.'})
            continue
        if abs(gap) <= deadband:
            ledger.append({**row, 'target': round(claim, 2), 'factor': 1.0,
                           'reason': f'Claim within {deadband:.1f} carries of the budget; left unadjusted.'})
            continue
        target = budget + np.sign(gap) * deadband
        if lead_weighted:
            w = full.loc[idx].clip(lower=0.0) ** 2
            new = (full.loc[idx] - (claim - target) * w / w.sum()).clip(lower=0.0)
            factor = (new / full.loc[idx].replace(0.0, np.nan)).fillna(1.0)
        else:
            factor = pd.Series(target / claim, index=idx)
        live = idx[~sidelined.loc[idx].to_numpy()]
        side = idx[sidelined.loc[idx].to_numpy()]
        for col in ('rushing_attempts',) + RUSH_DEPENDENTS:
            if col in out.columns:
                out.loc[live, col] = (pd.to_numeric(out.loc[live, col], errors='coerce').fillna(0.0)
                                      * factor.loc[live]).round(3)
        for col in ('_full_rushing_attempts',) + tuple(f'_full_{c}' for c in RUSH_DEPENDENTS):
            if col in out.columns and len(side):
                out.loc[side, col] = (pd.to_numeric(out.loc[side, col], errors='coerce').fillna(0.0)
                                      * factor.loc[side]).round(3)
        ledger.append({**row, 'target': round(float(target), 2),
                       'factor': round(float((full.loc[idx] * factor).sum() / claim), 4),
                       'reason': (f"Claim {claim:.1f} vs budget {budget:.1f}; room {'trimmed' if gap > 0 else 'raised'} "
                                  f"to {target:.1f} ({'lead-weighted' if lead_weighted else 'uniform'}).")})
    return out, pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
