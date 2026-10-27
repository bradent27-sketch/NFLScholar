# Season trend: does the league lean toward running as the season goes on? (2026-10-27)

User: "there is a chance that seasons start with higher passing volume and trend towards running volume as the season continues
as a slight lean... likely due in some part to weather".

**Short answer: yes, the lean is real. Outdoors it is mostly temperature. The board already follows it, so there is nothing to
build.** Two smaller in-season drifts on the board (RB carries in cold games, TE1 targets late in the season) fail out of sample and
do not show up in points. Script: `scripts/analyze_season_trend.py`; output in `.sweeps/season_trend_2026-10-27.txt`.

## 1. League (play-by-play 2020-2025, regular season, 3,230 team-games)

| Weeks | Pass attempts | Designed rushes | Neutral pass rate | Pass over expected | Outdoor temp (F) |
|---|---|---|---|---|---|
| 1-4 | 33.8 | 24.1 | 0.521 | -1.3 | 74 |
| 5-8 | 34.0 | 24.0 | 0.524 | -1.0 | 64 |
| 9-12 | 33.5 | 24.5 | 0.511 | -2.1 | 56 |
| 13-15 | 33.7 | 24.7 | 0.509 | -2.1 | 47 |
| 16-18 | 32.8 | 24.9 | 0.495 | -2.7 | 44 |

Neutral = win probability 0.2-0.8, quarters 1-3, downs 1-2, more than 2 minutes left in the half. The figures are per team-game.

- **Size.** Each week, pass attempts fall by 0.065 [-0.11, -0.02] and designed rushes rise by 0.068 [+0.04, +0.09]. Over a
  season that is about 1 pass becoming 1 run per team-game, about 3%. Neutral pass rate falls by 0.0018 a week. The rate is flat
  through week 8 and falls after that. Week 18 is the low point (31.8 attempts, neutral 0.484), but rested starters and teams with
  nothing to play for are mixed in there.
- **Not every season.** In 2020 neutral pass rate rose all year. In 2024 early passing was low
  (30.3 attempts in weeks 1-2).
- **Weather, outdoor games.** These are fitted with team-season fixed effects. The week slope on neutral pass rate is -0.0022.
  Adding temperature and wind cuts it to -0.0004 (se 0.0007).
  - Temperature: +0.00084 per degree F. 30 degrees colder is 2.5 points less neutral passing.
  - Wind: -0.0012 per mph.
  - Pass over expected behaves the same way (week -0.112 alone, -0.014 with weather).
  - **The play-calling lean outdoors is temperature.**
- **Indoor games** (dome or closed roof). Neutral pass rate barely moves (-0.0009 a week, interval touches 0). But pass attempts
  fall about as much as outdoors (-0.064 a week). Outdoors, temperature does not explain attempts either: +0.006 per degree
  (se 0.014). Wind does: -0.089 per mph (se 0.035).
  - Total pass volume also falls for reasons other than weather: late-season game scripts, backup QBs, week 18.
- **Efficiency.** Yards per dropback rise 0.0096 per degree (30 degrees colder is -0.29 yards, about -4.5%) and fall 0.042 per mph
  of wind. With weather in the fit, the week term goes to zero.

## 2. Does the board follow it? (harness base boards, 2022-2025 weeks 3-17)

Projected minus actual. In brackets: sum projected / sum actual.

| | Weeks 3-6 | 7-10 | 11-14 | 15-17 | Slope a week [90%] |
|---|---|---|---|---|---|
| Team pass attempts | -1.77 | -1.05 | -1.62 | -1.02 | +0.053 [-0.02, +0.13] |
| Team carries | -1.06 | -1.37 | -1.00 | -1.30 | -0.021 [-0.08, +0.04] |
| QB attempts (starters) | +0.71 (1.022) | +1.24 (1.040) | +0.03 (1.001) | +1.42 (1.046) | +0.014 [-0.07, +0.09] |
| QB points | +0.52 | +0.13 | +0.16 | +0.12 | -0.028 [-0.10, +0.04] |
| WR points | -0.15 | +0.40 | -0.17 | -0.05 | -0.002 [-0.04, +0.04] |
| RB carries | +0.07 (1.006) | -0.15 (0.986) | -0.22 (0.981) | -0.34 (0.972) | **-0.039 [-0.079, -0.005]** |
| RB points | -0.01 | -0.08 | -0.09 | -0.26 | -0.009 [-0.06, +0.04] |
| TE targets | -0.32 (0.938) | -0.59 (0.888) | -0.66 (0.874) | -0.72 (0.866) | **-0.036 [-0.062, -0.012]** |
| TE points | -0.62 | -0.85 | -0.23 | -0.89 | +0.005 [-0.05, +0.07] |

Player rows are played games with a projection of at least 5 points.

- **Passing: no drift.** Team-level attempts run low in every period. That is the known level gap from players who are not on the
  board, and it has no trend. QB attempts and points have no week trend and no temperature pattern either:
  - QB points by game temperature: indoor +0.16, above 60F -0.47, 46-60F +0.09, 33-45F +1.44, 32F and below -0.57.
  - WR is the same.
  - Recency weighting plus the wind feature already carry the passing side of the lean.
- **RB carries: a weather drift, but no points.** Outdoors the under-projection grows late and in the cold: carries ratio 1.024
  above 60F, 0.969 at 33-45F, 0.945 at 32F and below. Indoors it stays flat (week slope -0.005). The fast screen tried a cold-game
  carries lift, x(1 + k * max(0, T0 - temp)), fitted on two seasons and scored on the other two, four ways:
  - In two folds the fit picks no lift.
  - In the other two it picks a lift, and the lift hurts the held-out seasons (+0.018, +0.076 MSE).
  - RB points have no drift.
  - **Not a candidate.** This matches the 2026-09-01 weather rework, which dropped its cold-game terms.
- **TE targets: a seasonal drift, not weather.** League TE share of RB/WR/TE targets rises over the season: 0.203 in weeks 1-2,
  0.214 in weeks 3-6, then 0.222-0.227. Most of that comes out of the WR share (0.606 to 0.589). The board's TE share stays flat
  at about 0.20 (2024: league 0.205 to 0.244, board 0.195 to 0.207).
  - The drift sits in TE1s: ratio 0.901 to 0.823, slope -0.039 [-0.061, -0.015]. Within that it is established players (years 3-4
    -0.030, 5+ -0.020), not rookies (+0.005).
  - It is the same indoors and outdoors, and temperature adds nothing.
  - TE points have no drift: the 2024-2025 TE1 points bias shrinks late.
  - The fast screen tried a TE1 target lift that grows with the week. It helps 3 of 4 held-out folds slightly (-0.05 to -0.15 MSE
    on about 45) and hurts one (+0.26).
  - **Watch item W9, not a candidate.**

## 3. The 2026 link (W1)

- **Weeks 3-6 2026 were not the seasonal pattern.** Teams threw 34.5 and 35.3 a game, against 31.8 and 32.7 in 2025. The normal
  early-to-late gap is about 1 attempt; 2026 ran 2-3 above recent seasons.
- **The board caught up just as passing cooled.** Projected / actual team attempts by 2026 week: 0.905, 0.926, 0.939, 0.847, then
  **1.046 in week 7** (projected 33.0, actual 31.5; 26 teams, Monday's DAL-PHI not in the data yet).
- **Faster in-season volume weights (M2) would have hurt week 7.** The seasonal lean says passing keeps cooling from here, which
  also argues against chasing the early surge. W1 waits for week 8.

## Not covered

Weeks 1-2 and 18 are outside the harness window. Week 18 has the largest league shift (and rested starters). It would need its own
board run to check.
