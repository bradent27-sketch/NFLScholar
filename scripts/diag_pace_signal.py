"""
How the pace multiplier is built, what it gives a given game, and how well its signal predicts plays.
(2026-10-06, from the ATL @ NO question: NO runs the most plays a game, yet ATL players get x0.999.)

The model (data.weekly_projections, "pace_mult"): an offense's expected extra volume is the OPPOSING
DEFENSE's plays-faced per game against the league mean (load_team_pace 'def_pace' - what offenses
managed against that defense, NOT how fast that team's own offense plays), clipped to 0.85-1.15 and
shrunk toward 1 by games / (games + PACE_PRIOR_GAMES=12).

  part 1  the real numbers for one matchup (default 2026 wk5 ATL @ NO)
  part 2  historical test, 2016-2025 team-weeks: regress this game's plays minus the team's own to-date
          average on (a) the opponent defense's plays-faced vs league, (b) the opponent OFFENSE's own pace
          vs league, (c) own defense's plays-faced vs league, by games-played bucket. The coefficient on
          (a) is what the shrink factor ought to be; (b)/(c) are signals the multiplier ignores.

    python scripts/diag_pace_signal.py --team ATL --opp NO --year 2026 --week 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.loaders import _discount_overtime_plays, load_team_pace  # noqa: E402
from data.weekly_projections import PACE_CLIP, PACE_PRIOR_GAMES  # noqa: E402


def part1(team, opp, year, week):
    pace = load_team_pace(year, through_week=week)
    if pace is None or pace.empty:
        print('no pace data'); return
    league_def = pace['def_pace'].mean()
    ratio = float(np.clip(pace.loc[opp, 'def_pace'] / league_def, *PACE_CLIP))
    g = float(pace.loc[opp, 'def_games'])
    alpha = g / (g + PACE_PRIOR_GAMES)
    mult = 1.0 + alpha * (ratio - 1.0)
    pace = pace.assign(off_rank=pace['off_pace'].rank(ascending=False, method='min'),
                       def_rank=pace['def_pace'].rank(ascending=False, method='min'))
    print(f"\n=== {year} through week {week - 1}: {team} players vs {opp} ===")
    print(pace.loc[[opp, team], ['off_pace', 'off_rank', 'def_pace', 'def_rank', 'def_games']].round(1).to_string())
    print(f"league mean off/def plays per game: {pace['off_pace'].mean():.1f} / {league_def:.1f} "
          f"(range off {pace['off_pace'].min():.1f}-{pace['off_pace'].max():.1f}, def {pace['def_pace'].min():.1f}-{pace['def_pace'].max():.1f})")
    print(f"{opp} def_pace {pace.loc[opp, 'def_pace']:.1f} / league {league_def:.1f} = {pace.loc[opp, 'def_pace'] / league_def:.3f} "
          f"-> clip {ratio:.3f} -> shrink alpha = {g:.0f}/({g:.0f}+{PACE_PRIOR_GAMES:.0f}) = {alpha:.3f} -> multiplier {mult:.3f}")
    print(f"ceiling this week: a defense at the +15% clip would give x{1 + alpha * 0.15:.3f}; "
          f"after 12 games x{1 + 12 / 24 * 0.15:.3f}, a full 17 x{1 + 17 / 29 * 0.15:.3f}")
    print(f"{opp} OFFENSE is rank {int(pace.loc[opp, 'off_rank'])} in plays run, but the multiplier reads the plays its DEFENSE faced "
          f"(rank {int(pace.loc[opp, 'def_rank'])}).")


def team_weeks(years):
    import nflreadpy
    frames = []
    for year in years:
        try:
            df = nflreadpy.load_team_stats([year], summary_level='week').to_pandas()
        except Exception as exc:
            print(year, 'failed', exc); continue
        df = df[pd.to_numeric(df['week'], errors='coerce') <= 18].copy()
        if 'season_type' in df.columns:
            df = df[df['season_type'].astype(str).str.upper() == 'REG']
        cols = [c for c in ('attempts', 'carries', 'sacks_suffered') if c in df.columns]
        df['plays'] = _discount_overtime_plays(df['team'], df['week'], df[cols].sum(axis=1), year)
        df['year'] = year
        frames.append(df[['year', 'week', 'team', 'opponent_team', 'plays']])
    return pd.concat(frames, ignore_index=True)


def part2(years):
    tw = team_weeks(years)
    rows = []
    for year, d in tw.groupby('year'):
        d = d.sort_values('week')
        for week in sorted(d['week'].unique()):
            prior = d[d['week'] < week]
            if prior.empty:
                continue
            off = prior.groupby('team')['plays'].agg(['mean', 'count'])
            dfn = prior.groupby('opponent_team')['plays'].mean()
            m_off, m_def = off['mean'].mean(), dfn.mean()
            cur = d[d['week'] == week]
            for _, r in cur.iterrows():
                t, o = r['team'], r['opponent_team']
                if t not in off.index or o not in off.index or o not in dfn.index or t not in dfn.index:
                    continue
                rows.append({'year': year, 'week': week, 'games': off.loc[o, 'count'], 'plays': r['plays'],
                             'own_off': off.loc[t, 'mean'], 'x_opp_def': dfn[o] - m_def,
                             'x_opp_off': off.loc[o, 'mean'] - m_off, 'x_own_def': dfn[t] - m_def})
    r = pd.DataFrame(rows)
    r['dy'] = r['plays'] - r['own_off']
    print(f"\n=== plays per team-game, {min(years)}-{max(years)}: n={len(r)} ===")
    print(f"sd of a team's plays in a game around its own to-date mean: {r['dy'].std():.1f}")
    out = []
    for lab, (lo, hi) in {'2-4 games': (2, 4), '5-8': (5, 8), '9-12': (9, 12), '13-17': (13, 17)}.items():
        s = r[(r['games'] >= lo) & (r['games'] <= hi)]
        X = np.column_stack([np.ones(len(s)), s['x_opp_def'], s['x_opp_off'], s['x_own_def']])
        beta, *_ = np.linalg.lstsq(X, s['dy'].to_numpy(), rcond=None)
        resid = s['dy'].to_numpy() - X @ beta
        cov = np.linalg.inv(X.T @ X) * resid.var(ddof=4)
        se = np.sqrt(np.diag(cov))
        g = (lo + hi) / 2
        out.append({'bucket': lab, 'n': len(s), 'model alpha (n/(n+12))': round(g / (g + PACE_PRIOR_GAMES), 2),
                    'coef opp def plays-faced': f"{beta[1]:+.2f} ({se[1]:.2f})",
                    'coef opp OFFENSE pace': f"{beta[2]:+.2f} ({se[2]:.2f})",
                    'coef own defense faced': f"{beta[3]:+.2f} ({se[3]:.2f})",
                    'sd(x_opp_def)': round(s['x_opp_def'].std(), 2), 'sd(x_opp_off)': round(s['x_opp_off'].std(), 2)})
    print(pd.DataFrame(out).to_string(index=False))
    # how much of the variance of a game's plays does each signal explain, pooled 5-17 games
    s = r[r['games'] >= 5]
    for name, cols in (('opp def only (what the model reads)', ['x_opp_def']),
                       ('opp def + opp offense pace', ['x_opp_def', 'x_opp_off']),
                       ('all three', ['x_opp_def', 'x_opp_off', 'x_own_def'])):
        X = np.column_stack([np.ones(len(s))] + [s[c] for c in cols])
        beta, *_ = np.linalg.lstsq(X, s['dy'].to_numpy(), rcond=None)
        res = s['dy'].to_numpy() - X @ beta
        print(f"  {name:<38} R2 {1 - res.var() / s['dy'].var():.4f}  rmse {np.sqrt((res ** 2).mean()):.2f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', default='ATL')
    ap.add_argument('--opp', default='NO')
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--week', type=int, default=5)
    ap.add_argument('--years', default='2016-2025')
    a = ap.parse_args()
    part1(a.team, a.opp, a.year, a.week)
    y0, y1 = (int(v) for v in a.years.split('-'))
    part2(list(range(y0, y1 + 1)))


if __name__ == '__main__':
    main()
