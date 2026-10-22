# Model audit, 2026-10-21: where the error is, and what is still predictable

User request: screen for improvements more effectively than one setting at a time ("Come up with a plan on how to MOST
effectively audit our model with the data we have"), approved plan steps 1-3 plus scikit-learn and per-week harness output.

Why the approach changed: the calibration audit (C1-C16) re-tuned 17 settings that had been tuned before and shipped one
(C10, START-ALL -0.004). The larger wins this fall came from wrong inputs and missing situations (QB1 latest start -0.092, the
probable-starter fallback -0.028) and new information (expected-points luck -0.007, top-end yardage -0.008). So this audit starts
from the errors: where they come from, whether anything known before kickoff predicts them, and what the market knows that we do not.

Everything here is diagnostic. A lead becomes a named, mechanistic model flag with a pre-set bar and goes through the harness; the
market and FantasyPros stay benchmarks, never inputs.

**Data.** The current model's board (DEFAULT_FEATURES 60, the 2026-10-21 capture's board dump, verified identical to a plain build):
36,865 player-weeks, 2021-2025 weeks 3-17; harness START pool, played: 10,072 rows. Live: the 2026 ledger, weeks 3-6, each
player's last pre-kickoff board, paired with the market (574 START player-weeks) and FantasyPros (600).

**Scripts.** `scripts/audit_error_budget.py` (step 1), `scripts/audit_features.py` (107 pre-game features per board row),
`scripts/audit_residual_scan.py` (step 2), `scripts/audit_leads.py` (step 2 drill-down), `scripts/audit_market_gap.py` (step 3).
Outputs in `.sweeps/audit_*.txt`.

## Step 1: error budget

Each row's projected stat line is split into team opportunities x the player's share x per-opportunity rates; one piece at a time
is replaced by what happened and the row re-scored (one least-squares line per position for every column, so calibration is not a
handicap). The drop in squared error is the most that piece could ever be worth: an upper bound, not a forecastable gain.

| Share of squared error removed (START pool) | team volume | player share | opportunities | efficiency (non-TD) | TDs |
|---|---|---|---|---|---|
| QB | 0.038 | 0.084 | 0.140 | 0.352 | 0.495 |
| RB | 0.089 | 0.296 | 0.422 | 0.222 | 0.373 |
| WR | 0.057 | 0.320 | 0.417 | 0.254 | 0.238 |
| TE | 0.075 | 0.353 | 0.437 | 0.217 | 0.231 |
| all | 0.065 | 0.277 | 0.374 | 0.257 | 0.320 |

The same in every season (team 0.045-0.075, share 0.22-0.30, TDs 0.30-0.35).

Poisson floor (the share of each stat's squared error that pure Poisson noise around a perfect mean would leave): TDs 0.87-1.22
for every position; targets 0.64; receptions 0.68-0.73; RB carries 0.40; QB attempts 0.46; QB rush attempts 0.67.

Reading:
- **TDs are at their noise floor.** A third of the error, none of it reducible in the mean. No more TD-level work.
- **Team volume is worth at most 6.5% even if it were perfect.** Pace, pass rate and team play counts are not where the error is
  (matches the earlier nulls: M1, N6 PROE, C8 pace).
- **The reducible part lives in player opportunities:** the share of targets and carries a player gets (bound 28%), most of all RB
  carries (60% of their error above the Poisson floor) and WR/TE targets (36%).

## Step 2: residual scan (held out by season)

Residual = actual minus the board. A gradient-boosted model of the residual on all 107 pre-game features, fit on four seasons and
scored on the fifth, five times; START pool.

| Headroom (board + half the predicted residual) | n | base RMSE | dRMSE | CI | seasons better |
|---|---|---|---|---|---|
| QB | 1,726 | 7.642 | -0.018 | [-0.052, +0.016] | 4/5 |
| RB | 2,886 | 7.713 | -0.016 | [-0.043, +0.008] | 4/5 |
| WR | 4,003 | 7.861 | -0.041 | [-0.067, -0.014] | 5/5 |
| TE | 1,457 | 6.854 | -0.066 | [-0.095, -0.036] | 5/5 |
| START-ALL | 10,072 | 7.642 | **-0.033** | [-0.048, -0.018] | 5/5 |

At full strength it overfits (START-ALL +0.011); without the in-sample expected-points features the gain is -0.031. **With the data we
already have, about 0.03 START-ALL is still predictable** (about 1% of the squared error, eight times the C10 gain), concentrated in
WR and TE. The six leads below account for nearly all of it. Volume stats: targets -0.077 (5/5 seasons, 6% of the target error at
full strength), RB carries -0.109 (5/5); team totals: carries nothing (0/5), team targets -0.12 of 7.75.

Most important features on the held-out seasons (permutation): implied team total, last game's snap share, draft pick, last season's
target share, temperature, expected points to date, projected receiving TDs, the board minus the last-3-games average, age.
(Temperature and wind in nflverse's schedule are recorded game conditions, not a pre-game forecast; not used as a lead.)

Leads: one linear term in the feature, fit on four seasons and scored on the fifth, against the board (START pool):

| ID | Lead | Evidence (residual = actual - board) | Held out | ~START-ALL |
|---|---|---|---|---|
| L1 | **QB x implied team total** | QB residual rises from -1.6 points (lowest fifth of implied totals) to +1.2 (highest), 4-5 of 5 seasons per bin; also week to week against the team's own average (-1.1 to +1.2), so it is the game environment, not only team quality. It runs through passing TDs (-0.15 to +0.13) and rushing yards (-2.0 to +0.6). The fitted QB elasticities (passing TDs 0.27, rushing TDs 0.24, rushing yards 0.19) look too small. | START-QB -0.048 (4/5) | ~-0.008 |
| L2 | **WR target concentration** | WRs with a top-quarter target share last season are under-projected about +1.0 points all season (weeks 12-17 still +1.0, targets +0.95); the bottom quarter is over-projected -0.7 to -1.2 (4-5/5). It does not fade as the season's own data arrive, so the shares look too flat between the top receiver and the depth, not only too slow to update. | START-WR -0.025 (5/5); TE -0.012 (3/5) | ~-0.010 |
| L3 | **Rookie RBs and WRs** | RB rookies +1.3 to +1.9 points per START game (carries +1.7, rush yards +7.9, 5/5 seasons); WR rookies +1.4 to +2.2 (targets +0.9, yards +8.1, 5/5), larger late in the season. Second-year WRs run -1.0 in weeks 3-6 (5/5). | START-RB -0.016, START-WR -0.015 (4/5) | ~-0.010 |
| L4 | **TE level** | TE targets under-projected in every season (+0.4 to +0.8), yards +3.4 to +6.5; points +0.6, +1.0, +1.1 in 2023-2025 (watch item W3). | START-TE points-line refit -0.023 | ~-0.003 |
| L5 | RB game script | RBs on favored teams beat the board (spread). | START-RB -0.008 (5/5) | ~-0.002 |
| L6 | Last game's snap share | A jump in the last game's snaps over the 3-game average is under-read (RB, TE). | RB -0.010 (4/5), TE -0.007 (3/5) | ~-0.003 |

Nothing else among the 107 features carries a useful signal of its own. Coaching changes, rest, days since the last game,
opponent allowance, team change, QB change, sidelined teammates and PFF weekly coverage are each worth at most about 0.01 at one
position, mostly not in every season. The largest are QB home (-0.011, 4/5) and QB offensive-coordinator change (-0.004, 5/5).

### L2 mechanism: where the targets go (capture traces, every board row, actual 0 when not played)

Team target totals are right (31.97 projected vs 31.92 actual per team-week), and so is the split by within-team rank overall. The
flatness is inside each position room: TE1 +0.40 targets per game under (5/5 seasons) while TE2-TE4 are over (-0.15, -0.12, -0.09,
over in 5/5); WR1 +0.16 under (5/5) while WR2 is -0.17 over. Through the build stages:

| Mean targets per game | actual | blended rate x availability | after participation etc. | + vacancy | + pass capacity = final | miss at final |
|---|---|---|---|---|---|---|
| TE1 | 4.77 | 4.84 | 4.69 | +0.32 | -0.64 | +0.40 |
| TE2 | 1.50 | 2.04 | 1.78 | +0.12 | -0.25 | -0.15 |
| TE3 | 0.63 | 1.00 | 0.83 | +0.04 | -0.12 | -0.12 |
| WR1 | 7.44 | 7.89 | 7.85 | +0.47 | -1.05 | +0.16 |
| WR2 | 5.28 | 5.76 | 5.65 | +0.57 | -0.78 | -0.17 |
| WR3 | 3.22 | 3.69 | 3.45 | +0.35 | -0.50 | -0.08 |

Before the pass-capacity step the room leader is only a little over (TE1 5%, WR1 11%) and the backups much more (TE2 21%, TE3 27%,
WR2/WR3 15%). That comes from the backups' blended rates (last season's rate from a bigger role, TE3 1.83 per game against 0.63
actual) and from vacancy, which hands WR2 more than WR1. The capacity step then cuts every catcher in the group by the same ~12.5%,
which over-cuts the leader and under-cuts the depth. The uniform cut is the user's 2026-08-30 choice: it replaced a version that
protected WR1/WR2/TE1, which the user called "not correct", and the allocator's own comment says trimming a reserve is the role
model's job upstream. So the fix candidates are upstream: the backups' prior and participation, and the vacancy split by rank (the
09-25 rank-decay sweep already found the shipped decay too weak). A rank-aware capacity cut needs the user's call first. TE1
under-projection is most of the TE level gap (L4).

## Step 3: market gap (live 2026, weeks 3-6)

| START pool, paired | n | model RMSE | market RMSE | info slope |
|---|---|---|---|---|
| QB | 99 | 6.65 | 6.62 | 0.54 |
| RB | 145 | 7.07 | 6.97 | 0.64 |
| WR | 239 | 7.80 | 7.33 | 1.04 |
| TE | 91 | 7.54 | 7.33 | 0.95 |
| all | 574 | 7.39 | 7.12 | 0.87 |

Info slope = (actual - model) regressed on (market - model): 0 means the disagreement carries no information, 1 means the market is
right whenever the two disagree. On WRs and TEs it is about 1. FantasyPros gives the same picture (WR 1.00, TE 1.37, all 0.86).

- **By stat** (the model's points with one stat taken from the market): WR receptions -0.25, WR receiving yards -0.23, WR TDs -0.13,
  TE yards -0.11, QB passing yards -0.03 (FantasyPros: QB passing yards -0.11, passing TDs -0.13). The market's edge is WR/TE volume,
  which is lead L2's territory.
- **Our large disagreements are wrong on average in both directions:** where the model is 3+ points above the market it was 3.4 too
  high; 3+ below, 4.6 too low.
- Of the 40 largest weekly disagreements the market was closer on 25. Some are news timing (Chase and McLaurin projected 0 on the last
  pre-kickoff board and played: D2); the model was right on Bagent (QB1) and A.J. Brown.
- Live QBs are under-projected on passing volume (passing yards -9 to -12, attempts -2.5 to -2.9; FantasyPros -0.4): watch item W1.

Most of the 0.27 live gap is information the backtest features do not contain (only ~0.03 of error is predictable from them). Past
L1-L6, the next gains need new information, not tuning. Caveat: four weeks; the CIs come from four weekly clusters and are not
reliable yet. Re-run `scripts/audit_market_gap.py` after each week's games.

## Next

1. Build L2, L1 and L3 as named flags, in that order (L2 is the largest and sits where the market gap is). Each gets a pre-set bar,
   the fast screen (step 5, being validated) and then the full harness. L4 is a points-line refit (cheap, calibration input).
2. Each week: `scripts/audit_market_gap.py` (market and FantasyPros) and the big-miss read.
3. Fast screen (step 5): 20 fixed weeks (4, 7, 10, 13, 16 of 2022-2025), validation run on eight flags with known full-harness results.
   The harness now writes per-week paired metrics (`.sweeps/harness_weekly/`), so later runs can check how often the screen agrees.
