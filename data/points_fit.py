"""
A two-way points fit for the Weekly Rankings GAMES view (2026-10-23; the counterpart of CFB Scholar's points_ratings /
model_game_points, which fill the same "MODEL vs MARKET" panel there).

Each team gets an offense and a defense rating on points, fit jointly on game scores so a rating is opponent-adjusted:

    points(team vs opponent) = league mean + offense[team] + defense[opponent] + home field (home side only)

by weighted ridge least squares - this season's regular-season games before the week at full weight, last season's at
PRIOR_SEASON_WEIGHT, every rating pulled toward 0 (league average) by RIDGE_GAMES games' worth of evidence, so a team two
games in is mostly last season and the league mean.

DISPLAY ONLY. No projection reads it: the board's game script and implied totals are the market's. The game card sets
this independent read of the score beside the market's, the way CFB Scholar's does.
"""
import numpy as np
import pandas as pd

PRIOR_SEASON_WEIGHT = 0.35
RIDGE_GAMES = 4.0


def _games(schedule, weight):
    if schedule is None or schedule.empty:
        return pd.DataFrame()
    g = schedule.copy()
    if 'game_type' in g.columns:
        g = g[g['game_type'].astype(str) == 'REG']
    g = g.dropna(subset=['home_score', 'away_score'])
    if g.empty:
        return pd.DataFrame()
    neutral = g['location'].astype(str).str.lower().eq('neutral') if 'location' in g.columns else False
    home = pd.DataFrame({'offense': g['home_team'].astype(str), 'defense': g['away_team'].astype(str),
                         'points': pd.to_numeric(g['home_score'], errors='coerce'), 'home': (~neutral).astype(float)})
    away = pd.DataFrame({'offense': g['away_team'].astype(str), 'defense': g['home_team'].astype(str),
                         'points': pd.to_numeric(g['away_score'], errors='coerce'), 'home': 0.0})
    out = pd.concat([home, away], ignore_index=True).dropna(subset=['points'])
    out['weight'] = float(weight)
    return out


class PointsFit:
    """mu, home field, and {team: rating} for offense and defense (0 = league average)."""

    def __init__(self, mu, home, offense, defense, games):
        self.mu, self.home, self.offense, self.defense, self.games = mu, home, offense, defense, games

    def expected(self, team, opponent, is_home, neutral=False):
        """The points `team` scores against `opponent`."""
        return float(self.mu + self.offense.get(team, 0.0) + self.defense.get(opponent, 0.0)
                     + (self.home if is_home and not neutral else 0.0))

    def game(self, home_team, away_team, neutral=False):
        """(home points, away points)."""
        return (self.expected(home_team, away_team, True, neutral), self.expected(away_team, home_team, False, neutral))


def fit_points(current_schedule, week, prior_schedule=None, prior_weight=PRIOR_SEASON_WEIGHT, ridge_games=RIDGE_GAMES):
    """The fit from this season's games before `week` plus last season's, or None when there are no scores at all."""
    current = current_schedule
    if current is not None and not current.empty and 'week' in current.columns:
        current = current[pd.to_numeric(current['week'], errors='coerce') < int(week)]
    rows = pd.concat([_games(current, 1.0), _games(prior_schedule, prior_weight)], ignore_index=True)
    if rows.empty:
        return None
    teams = sorted(set(rows['offense']) | set(rows['defense']))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # columns: mu, home, offense[n], defense[n]
    x = np.zeros((len(rows), 2 + 2 * n))
    x[:, 0] = 1.0
    x[:, 1] = rows['home'].to_numpy(dtype=float)
    x[np.arange(len(rows)), 2 + rows['offense'].map(index).to_numpy()] = 1.0
    x[np.arange(len(rows)), 2 + n + rows['defense'].map(index).to_numpy()] = 1.0
    w = rows['weight'].to_numpy(dtype=float)
    y = rows['points'].to_numpy(dtype=float)
    penalty = np.zeros(2 + 2 * n)
    penalty[2:] = float(ridge_games)
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x + np.diag(penalty), xtw @ y)
    offense = {t: float(beta[2 + i]) for t, i in index.items()}
    defense = {t: float(beta[2 + n + i]) for t, i in index.items()}
    return PointsFit(float(beta[0]), float(beta[1]), offense, defense, int((rows['weight'] == 1.0).sum() // 2))


def week_points_fit(year, week):
    """fit_points for one board week from the app's schedules; None when it cannot be built."""
    from data.loaders import load_schedule
    try:
        current = load_schedule(int(year))
    except Exception:
        current = None
    try:
        prior = load_schedule(int(year) - 1)
    except Exception:
        prior = None
    try:
        return fit_points(current, week, prior)
    except Exception:
        return None
