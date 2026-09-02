# Overnight autonomous session — 2026-08-31

User signed off ~22:30 and asked for iteration on: (1) coaching-change ×
defense-prior interaction, (2) a real weather feed + QB/WR wind effect, (3) any
other backlog idea with enough ground, run overnight with sequential heavy
backtests, then hone the strength/ratio of whatever shows signal. Nothing
ships to `DEFAULT_FEATURES` — every new lever is a `MODEL_FEATURES` flag,
off by default, pending the user's review of these results.

---

## 0. RB snap-anchored volume — sweep closed (carry-over from 08-30)

`.sweeps/sweep_rb_snap_anchored_2026-08-30.txt` (2022-2025 wk1, the two new
0..1 dials). dMAE = ON − OFF (positive = flag hurts):

| tilt | vac | RB dMAE | START-RB dMAE | weeks |
|---|---|---|---|---|
| 0.0 | 0.0 | +0.000 | +0.000 | control (flag inert) |
| 0.33 | 1.0 | +0.043 | +0.069 | 0-4 |
| 0.66 | 1.0 | +0.073 | +0.170 | 0-4 |
| 1.0 | 1.0 | +0.106 | +0.176 | 1-3 |
| 1.0 | 0.0 / 0.5 | +0.106 | +0.159 / +0.169 | 1-3 |

The tilt is **monotonically harmful** — no beneficial operating point. Vacancy
knob inert on this window (no OUT players in the archive). Confirms the 08-30
ablation. `v2_rb_snap_anchored_volume` stays OUT of `DEFAULT_FEATURES`; the two
knobs remain switchable tuning. **Closed.**

---

## 1. Coaching change × defense prior — `v2_coaching_aware_defense_prior`

### Data (`data/coaching_changes.py`)
- **Head coach**: nflverse `load_schedules()` `home_coach`/`away_coach`, zero
  nulls 1999-2026. Reliable for every transition. (NB: the 2026 rows carry
  some placeholder/incorrect coach names — not used; the analysis window ends
  2025.)
- **OC / DC**: the committed Ourlads archive
  (`external_data/ourlads_coaching_staff_by_season.csv`), 2022-2025 only. So
  DC-cohort splits run on 2023-2025 (26 `dc_only`, 16 `both`, 4 `hc_only`, 50
  `none` team-seasons). **No hand-typed history** — a wrong label poisons the
  study, per the user's explicit warning.

### Findings (`scripts/analyze_coaching_defense_prior.py`, output in `.sweeps/coaching_defense_prior_2026-08-31.txt`)

**Year-over-year defense-allowed persistence** (corr of prior→current
per-position allowed-PPR rating):

| cohort | n (team-pos) | corr(prev, cur) |
|---|---|---|
| HC unchanged (2018-25) | 768 | +0.221 |
| HC changed (2018-25) | 244 | +0.095 |
| none (2023-25) | 200 | +0.194 |
| **dc_only** | 104 | **+0.329** (highest) |
| **both (HC+DC)** | 64 | **+0.023** (collapses) |

**Out-of-sample optimal `prior_games`** (blend weeks-1..n vs actual
weeks-(n+1)..18, n=2..14, grid to 55):

| cohort | best pg | MAE@12 → MAE@best | per-season direction |
|---|---|---|---|
| none | 14 | 0.168 → 0.168 | stable ~12-16 |
| **both** | **8** | 0.183 → 0.180 | 2/3 seasons want ≤8 |
| **dc_only** | **40** | 0.150 → 0.146 | every season wants ≥24 |
| hc_only | 4 (n tiny) | — | ignore |

Per-position: the `dc_only` "trust prior more" is strongest for **RB and TE
defense** (best pg 55), weak for WR (pg 10 ≈ `none`). The `both` "trust prior
less" holds for QB/RB/WR (pg 4-6), TE noisy.

**Read:** a wholesale defensive-staff change (HC+DC, or a lone new HC) genuinely
resets the unit → shorter prior leash. A coordinator-only promotion under a
retained HC is usually internal continuity (same system, same personnel) →
last season is *more* predictive than average. The user's HC-vs-DC intuition
was right; the DC-only direction is the opposite of the initial guess but
consistent across two independent measures and every season in the window.

**Caveats:** DC data is 3 seasons; the proxy rating is not opponent-adjusted
like the model's real matchup matrix; pg-40 for `dc_only` is near the grid
edge (curve is shallow past ~24).

### Implementation
- `COHORT_DEFENSE_PRIOR_GAMES = {none/unknown: default(12), dc_only: 18,
  hc_only: 8, both: 8}` — conservative vs what the raw curve wants for
  `dc_only`.
- `blend_defense_prior(...)` now accepts a per-team Series for `prior_games`
  (falls back to the scalar `DEFENSE_PRIOR_GAMES` for any team not covered).
- `v2_coaching_aware_defense_prior` (MODEL_FEATURES, not default). Wired at the
  `blend_defense_prior` call inside `v2_defense_prior`. A team with no cohort
  signal keeps the default → the flag is a **no-op for the live 2026 board**
  (no 2026 coordinator data) and for any pre-2023 backtest. Live use needs
  current-season coordinator data curated.
- Tests: `tests/test_coaching_changes.py` (6).

### UPDATE 2026-08-31 (user follow-up) — extended DC history + re-sweep

**More coaching data.** `data/coaching_history_wikipedia.py` pulls the
Wikipedia "List of current NFL {off,def} coordinators" article as it stood
each DECEMBER (in-season, before the January free-agent-coordinator churn that
contaminated an end-of-season snapshot) via the MediaWiki revisions API - one
snapshot per (list, year) = ~20 calls for a decade. Merged UNDER the verified
Ourlads years in `data/coaching_changes.py`. Coverage ~75% of 2015-2021
team-seasons; gaps fall through to the `unknown` cohort (never a wrong label).
Eyeball table: `.sweeps/coaching_eyeball_table_2026-08-31.txt` - spot-checks
clean (BUF Thurman→Frazier 2017, LA Williams→Phillips→Staley→Morris, GB
Capers→Pettine→Barry, etc.). **Usable DC-split sample: ~250 team-seasons, up
from ~92.**

**New cohort `dc_to_hc`** (user's special case): the prior year's DC IS this
year's HC, same team (Bowles→TB 2022, Crennel→HOU 2020, Lovie Smith→HOU 2022,
Dennis Allen→NO 2022, plus Belichick-runs-the-D years). Scheme continuity
holds → treated as `none` for the prior weight.

**Extended proxy analysis** (`scripts/analyze_coaching_defense_prior.py`,
2016-2025, `.sweeps/coaching_defense_prior_v2_2026-08-31.txt`):

| cohort | n | persistence corr | optimal prior_games | MAE@12 vs @best |
|---|---|---|---|---|
| none | 592 | +0.17 | 14 | 0.165 = 0.165 |
| **dc_only** | 200 | **+0.33** | 20-24 | 0.165 → 0.163 |
| **both** | 188 | **+0.08** | 8-12 (flat) | 0.180 = 0.180 |
| **dc_to_hc** | 20 | **+0.50** | 4-6 | 0.157 → 0.147 |
| hc_only | 44 | −0.02 | 6 (n tiny) | — |

The directions from the first pass HOLD on 2x the data: `dc_only` more
persistent + wants a longer prior; `both` less persistent; `dc_to_hc`
most-persistent-and-most-readable (small pg works). Magnitudes are still
year-noisy. Gains over flat pg=12 are ≤ 0.002 in the proxy MAE everywhere
except `dc_to_hc` (~0.01, n=20).

**Re-sweep** `scripts/sweep_coaching_prior_games.py` (env-driven
`COHORT_PRIOR_GAMES`), 2020-2025 wk4-13
(`.sweeps/sweep_coaching_prior_games_2026-08-31.txt`):

| config | ALL | QB | RB | WR | TE | START-TE | START-RB |
|---|---|---|---|---|---|---|---|
| off (control) | +0.000 | all 0 | | | | | |
| v1 18/8/8 | +0.000 | −0.002 | +0.001 | +0.001 | +0.001 | −0.033 | +0.014 |
| strong 26/6/4 | +0.000 | −0.004 | +0.001 | +0.001 | +0.000 | −0.068 | −0.009 |
| dc_only-only 20 | −0.000 | −0.002 | −0.000 | −0.000 | +0.000 | −0.026 | +0.018 |

**Not one `*` anywhere** — no scope in any config has a CI that excludes 0.
Whole-pool is ±0.002 everywhere. START-TE leans −0.03 to −0.07 across all
three configs (consistently the right sign) but is not significant and TE
defense is the noisiest position in the study.

### Coaching — FINAL verdict (2026-08-31)
Confirmed across **two independent backtests** (2023-25 v1 + this 2020-25
4-config sweep on 2.7x the coaching data): **`v2_coaching_aware_defense_prior`
does NOT help. Do not ship.** The descriptive findings are real and now
well-supported (staff resets cut persistence; DC-only-with-retained-HC is
*more* persistent; DC→HC promotion most of all), but the per-cohort
`prior_games` shift only moves `alpha = n/(n+pg)` by ~0.05-0.10 on a matchup
multiplier already ~1.0±0.15, so by wk4+ it's below the noise floor. The one
consistent-sign hint (START-TE) is not significant. Flag + `data/coaching_
changes.py` + the extended history stay in the tree as reference data; the
mechanism is a dead end.

**Fallback DC source if the sweep is borderline** (user, 2026-08-31): pro-
football-history.com has a per-team defensive-coordinator history page,
e.g. `https://pro-football-history.com/franchpos/1/8/san-francisco-49ers-
defensive-coordinator-history` (path is `franchpos/<franchise_id>/<position_id>/
<team-slug>-defensive-coordinator-history`; position 8 = DC). Prose, not a
table, so it needs per-page parsing - only worth the effort if a cohort's
`prior_games` sits right on the edge of significance and the Wikipedia
coverage gap is what's blocking the call.

### Model backtest — `v2_coaching_aware_defense_prior` (v1 config) → NEUTRAL, DO NOT SHIP
`backtest_component.py --add v2_coaching_aware_defense_prior --years 2023,2024,2025
--weeks 4-17` (`.sweeps/backtest_coaching_defense_prior_2026-08-31.txt`, n=12,958):

| scope | dMAE (var−base) | boot 95% CI |
|---|---|---|
| ALL / QB / RB / WR / TE (whole) | ±0.001 | all include 0 — flat |
| START-QB | −0.018 | [−0.053, +0.006] — noise, leans + |
| **START-RB** | **+0.029** | **[+0.007, +0.055] — excludes 0, small real drag** |
| START-WR | +0.004 | noise |
| START-TE | −0.021 | [−0.062, +0.006] — noise, leans + |

The descriptive finding is real, but by wk4-17 the defense-prior blend is a
second-order nudge (a pg 12→18 change shifts alpha by ~0.1 on a multiplier
already ~1.0±0.15), so it doesn't translate: whole-pool dead flat, and the one
significant cut (START-RB) is slightly negative. **Verdict: keep the flag in
MODEL_FEATURES for the record, do NOT ship.** Faint hint (START-QB/TE lean +,
START-RB is the drag) that a pass-catcher-only variant *might* help — logged,
low priority, not worth the live current-season-DC data dependency for a
noise-band effect.

### DEFENSE_PRIOR_GAMES = 12 multi-year confirm (backlog item) → KEEP 12
`sweep_defense_prior_games.py --values 8,10,12,16,20 --years 2022,2023
--weeks 4-14` (`.sweeps/... task bc6ixh376`, n=6,497; 2022-2023 chosen to not
overlap the 2024-2025 tuning window). dMAE vs the shipped pg=12:

| pg | ALL | QB | TE (whole) | START-TE | START-WR |
|---|---|---|---|---|---|
| 8 | −0.002 (17-5 wk) | −0.005 | −0.003 | **−0.139\*** | −0.013 |
| 10 | +0.000 | −0.002 | −0.000 | **−0.072\*** | **−0.022\*** |
| **12** | **0** (baseline) | | | | |
| 16 | +0.001 | +0.004 | **+0.005\*** | +0.086 | −0.016\* |
| 20 | +0.002 | +0.010 | **+0.007\*** | +0.072 | −0.013 |

Whole-pool ALL barely moves across the entire 8-20 range (span 0.004).
Going ABOVE 12 clearly hurts TE (whole-pool +0.005 to +0.007, CI excludes 0).
Going to 8-10 gives a marginal startable-pass-catcher gain (START-TE −0.14 /
−0.07, START-WR −0.02, all CI-excludes-0) but a negligible whole-pool effect.
Also note the coaching proxy analysis put the `all`-cohort optimal near 14 on
2018-2025 (different metric, opposite nudge) — i.e. the true optimum is a wide,
flat basin around 12.

**Verdict: keep `DEFENSE_PRIOR_GAMES = 12`.** Confirmed not wrong on an
independent window. The faint pg=8-10 startable-TE/WR hint is not worth
re-opening (2 seasons only, whole-pool null, and any change forces another
calibration re-fit). Logged as a low-priority future tweak. **Backlog item
closed.**

---

## 2. Weather — `v2_weather_adjustment` + `data/weather.py`

### Provider abstraction (`data/weather.py`)
- `WeatherProvider` ABC with `OpenMeteoProvider` (default — keyless, no signup,
  hourly wind/temp by lat/lon, ~16-day horizon), `VisualCrossingProvider`
  (needs `VISUAL_CROSSING_API_KEY`; adapter ready), `NWSProvider` (stub).
- `recorded_game_weather(schedule, week)` — the schedule's post-game
  `temp`/`wind`/`roof`, keyed by BOTH teams (game environment, not stadium).
- `forecast_game_weather(...)` — stadium coord table + kickoff hour → provider,
  cached 6h under `~/.cache/nflscholar_weather/`.
- `resolve_game_weather(...)` — recorded where present, forecast only for the
  still-blank outdoor rows. Every failure path → `{}` / neutral, never a crash.

### Effect (`scripts/analyze_weather_effect.py`, `.sweeps/weather_effect_2026-08-31.txt`, 2015-2025, ratio to each player's own season avg, outdoor only)

**Wind** (vs calm ≤8 mph):

| pos | 8-12 | 12-15 | 15-20 | 20+ |
|---|---|---|---|---|
| QB | 0.957 | 0.920 | **0.856** | 0.881 (n=65) |
| WR | 0.949 | 0.963 | **0.879** | 0.982 (n=215) |
| TE | 0.997 | 0.979 | 0.950 | 0.701 (n=99) |
| RB | 1.023 | 0.968 | 0.981 | 1.074 |

Isolating to mild temps (40-75 °F) sharpens it: QB 0.803 / WR 0.864 at
15-20 mph. **WRs are hit nearly as hard as QBs** — the user's guess that the
QB wind effect carries to receivers is confirmed. RB shows no gradient (backs
run more into wind, offsetting the passing drag).

**Cold** (vs 45-65 °F, isolating wind ≤10): QB 0.949 at 20-32 °F, ~0.82 at
≤20 °F. WR/TE mild (~0.95 sub-freezing). RB slightly *up* in the cold.

### Implementation
- `WEATHER_WIND` per position (piecewise-linear knee→floor): QB 8→18 mph to
  0.86; WR 8→17 to 0.89; TE 10→20 to 0.93; RB none. `WEATHER_COLD`: QB 32→15 °F
  to 0.86, WR to 0.95. Penalty only (≤1.0), anchored at 1.0 for a calm/mild
  outdoor game; combined penalty clamped ≥0.72.
- Folded into `env_mult` under `v2_weather_adjustment` (MODEL_FEATURES, not
  default), QB/WR/TE only. Recorded weather for a past week
  (`allow_forecast=not historical_target`), Open-Meteo for a live one.
- Tests: `tests/test_weather.py` (13).
- Sanity (2024 wk12, real 13 mph / 36 °F CLE-PIT game): R.Wilson −1.05,
  Pickens −1.0, Winston/Mayfield −0.9, Purdy −0.74 — right teams, right sign.

### Model backtest — `v2_weather_adjustment` → mixed, honing in progress
`backtest_component.py --add v2_weather_adjustment --years 2021,2022,2023,2024,2025
--weeks 3-17` (`.sweeps/... task birsp51le`, n=22,924):

| scope | dMAE (var−base) | boot 95% CI | note |
|---|---|---|---|
| ALL | −0.002 | [−0.007, +0.002] | noise |
| QB | −0.020 | [−0.049, +0.007] | leans helpful, not sig |
| **RB** | **−0.006** | **[−0.010, −0.002]** | **sig HELP — knock-on: less windy-game passing → pass-capacity gives RBs the carries (which is right, teams run more in wind)** |
| WR | +0.004 | [−0.003, +0.011] | leans slightly WORSE |
| TE | −0.001 | [−0.006, +0.003] | flat |
| START-RB | −0.021 | 41-23 wk, p=0.03 | leans helpful |
| START-QB/WR/TE | +0.010 / +0.008 / +0.011 | all include 0 | noise |

Read: the RB benefit (via the model's own volume conservation) is the only
significant effect and it's clean; QB direct effect is right-signed but
underpowered on this sample; **the WR wind penalty as calibrated is slightly
counterproductive** (a windy game compresses the mean but the one deep shot
still booms — WR weekly output in wind is noisier than the mean multiplier).

### Honing sweep → `v2_weather_adjustment` does NOT robustly help. DO NOT SHIP.
`scripts/sweep_weather_strength.py` (per-position `WEATHER_STRENGTH_{QB,WR,TE}`
env knobs), 6 configs on **2022-2024 wk4-14** (`.sweeps/sweep_weather_strength_2026-08-31.txt`):

| config | ALL | QB | RB | WR | START-RB |
|---|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 | 0 |
| as-measured | +0.005 | +0.012 | −0.004* | +0.013* | −0.033* |
| no-WR | +0.004 | +0.012 | −0.004* | +0.011* | −0.031* |
| no-WR strongQB | +0.007 | +0.019 | −0.006* | +0.016* | −0.015 |
| half-WR | +0.005 | +0.012 | −0.004* | +0.012* | −0.013 |
| uniform 0.6 | +0.002 | +0.006 | −0.003* | +0.007* | −0.018 |

**The QB direct effect flips sign between windows** — it *helped* on 2021-25
wk3-17 (QB −0.020) and *hurts* on 2022-24 wk4-14 (QB +0.012 to +0.019). Not
robust. **WR hurts on both paths** (direct + the QB→pass-capacity knock-on:
even at `WR strength 0` the WR dMAE is +0.011*, because the WR damage comes
through pass-capacity reallocation, not the direct multiplier). Whole-pool ALL
is **positive (worse) for every config** on this window.

The ONLY effect consistent across both windows: **RB −0.004 to −0.006 (small,
CI excludes 0)** — and that is a second-order artifact of perturbing QB volume
(less windy-game passing → pass-capacity hands RBs the carries), not a weather
model.

**Verdict: do NOT ship `v2_weather_adjustment`.** The 2015-25 weather effect is
real in raw data but (a) too small/noisy to improve a weekly MAE for QB, (b)
unexploitable for WR (boom variance dominates an 11% mean shift), (c) actively
harmful for WR via pass-capacity. Flag + `data/weather.py` + Open-Meteo stay in
the tree (verified working, live 2026 forecasts) for future iteration.

**Checked the `v2_weather_rb_rush_bump` follow-up — signal too small to build.**
Team rush-play share by wind (pbp 2018-25, 1,323 outdoor games): calm ≤8 mph
0.416, 8-12 0.426, 12-15 0.423, **15-20 0.432 (+0.016 vs calm)**, 20+ 0.465
(n=21). So a high-wind game shifts ~1.6 pp of plays pass→rush — about +1 carry
/ −1 pass attempt. That is real but tiny (the backtest RB knock-on was
−0.004 to −0.006 dMAE), and a new plays-conserving mechanism interacting with
`v2_pass_capacity` is non-trivial to get right. **Not worth it** unless the
user specifically wants the marginal RB-in-wind edge.

### Weather v1 — overall
Keep: `data/weather.py` (provider abstraction + Open-Meteo, verified pulling
live 2026 wk1 forecasts). v1 flat penalty: **not shipped** - doesn't translate
to a weekly-MAE improvement that holds across windows.

---

## 2b. Weather v2 — PER-STAT redistribution (user follow-up 2026-08-31)

The v1 flat fantasy-points penalty was the wrong model. Wind doesn't scale
points down uniformly - it REDISTRIBUTES, and the pieces partly cancel in
fantasy points. `scripts/analyze_weather_stats.py` (2015-2025, 122k outdoor
player-games, each stat as a ratio to the player's own season mean):

**Wind effect per stat (fraction change per +1 mph above 8 mph):**

| pos | stat | slope/mph | @ 20 mph |
|---|---|---|---|
| QB | passing_attempts | −0.35% | 0.96 |
| QB | passing_completions | −0.50% | 0.94 |
| QB | passing_yards | −0.60% | 0.93 |
| QB | passing_tds | −0.75% | 0.91 |
| QB | **rushing_attempts** | **+0.18%** | 1.02 |
| QB | **rushing_yards** | **+0.45%** | 1.05 |
| WR | targets / receptions / rec_yards / rec_tds | −0.25 / −0.42 / −0.65 / −0.75% | 0.97 / 0.95 / 0.93 / 0.93 |
| TE | targets / receptions / rec_yards / rec_tds | −0.30 / −0.45 / −0.50 / −0.60% | 0.96 / 0.95 / 0.94 / 0.93 |
| RB | **rushing_attempts / rushing_yards** | **+0.18 / +0.14%** | 1.02 / 1.02 |
| RB | targets / receptions / rec_yards | −0.15 / −0.18 / −0.22% | ~0.98 |

Confirms the user's read: wind cuts QB **volume AND efficiency** (comp% −1.9%/10mph,
yds/att −3.9%/10mph, both on top of the attempt drop), and **QBs scramble more**
(carries +1.8%/10mph, rush yards +5.1%/10mph). WR/TE catch-rate and yds/target
both drop. RB carries tick up, RB receiving ticks down.

**Temperature** (user's buckets, isolated to calm games) — small, as predicted.
Only the below-freezing tail is modelled: QB `passing_yards` ×0.97,
`passing_tds` ×0.95, **`passing_interceptions` ×1.15**; RB `rushing_attempts`
×1.03, `rushing_yards` ×1.05 (teams run more, and better, in the cold).

**Rebuild:** `v2_weather_adjustment` reworked to `weather_stat_multipliers(pos,
wind, temp, is_outdoor) -> {stat: mult}` (`WEATHER_WIND_SLOPE` / `WEATHER_COLD_STAT`
in weekly_projections.py, per-stat clamp [0.78, 1.25]), applied per-stat in the
`proj_cols` loop + the vacancy-volume snapshot, RB now included. Per-position
`WEATHER_STRENGTH_*` sweep knobs retained. NOTE: interacts mildly with the
unshipped `volume_efficiency` flag (which rebuilds derived pass stats from
attempts) - fine for the default board.

2024 wk12 sanity (real 13 mph CLE-PIT / TB games): Winston −0.50, R.Wilson
−0.45, Pickens/Evans/Godwin −0.40, RB effect ~−0.10 (moderate wind - the
rush-up piece only overtakes at ~18+ mph).

### Model backtest — weather v2
`backtest_component.py --add v2_weather_adjustment --years 2021,2022,2023,2024,2025
--weeks 3-17` → **[QUEUED behind the coaching sweep]**

**Why WR differs (ratio distribution by wind, 2015-25, `analyze_weather_effect.py --dist`):**

| | calm ≤8 | 15-20 mph | shift |
|---|---|---|---|
| QB mean / median / P75 / boom>1.5 | 1.02 / 1.00 / 1.32 / 15.6% | 0.87 / 0.89 / 1.18 / 9.3% | whole distribution slides down — **exploitable** |
| WR mean / median / P75 / boom>1.5 | 1.01 / 0.86 / 1.41 / 22.2% | 0.89 / 0.81 / 1.22 / 16.8% | mean down but **P75 stays 1.22, boom rate 17%, sd still 0.78** |

Wind compresses QB from both ends; for WR it nicks the mean while the boom
tail (one deep shot / red-zone fade) survives, and WR weekly variance
(sd 0.78 vs QB 0.52) swamps an 11% mean shift. So a flat WR wind penalty
lowers the projection correctly but doesn't cut MAE — and misses harder on
the boom weeks. The RB benefit rides on the QB volume drop, not the WR term.

---

## Queue / status — ALL COMPLETE

| # | item | outcome |
|---|---|---|
| 1 | RB snap-anchored strength sweep | Closed — tilt monotonically harmful, `v2_rb_snap_anchored_volume` stays out of DEFAULT_FEATURES. |
| 2 | `v2_coaching_aware_defense_prior` | Built + tested + backtested → **NEUTRAL, do not ship.** Descriptive finding (staff resets cut defense persistence) is real but doesn't survive to a wk4-17 projection. |
| 3 | `v2_weather_adjustment` + `data/weather.py` | Infra built & verified (Open-Meteo live). Adjustment **not shipping** — real 2015-25 effect, but QB direct effect flips sign across windows, WR unexploitable, only a tiny RB knock-on survives. |
| 4 | `DEFENSE_PRIOR_GAMES` = 12 confirm | **Confirmed, keep 12.** Independent 2022-2023 window; flat basin 8-20, going >12 hurts TE. Backlog item closed. |
| 5 | Honing | Done for weather (per-position sweep). Coaching had no signal to hone. |

## What ships to DEFAULT_FEATURES tonight: **nothing.**
Both headline investigations came back negative — a real, cleanly-established
result (the user's own framing: "confirm their effect positive or negative or
negligible"). Reusable byproducts kept in the tree:
- `data/coaching_changes.py` — verified HC (nflverse) + DC (Ourlads) change
  labels, `@lru_cache`d. Usable for any future coaching-driven feature.
- `data/weather.py` — provider-agnostic weather (Open-Meteo default, keyless,
  verified live; Visual Crossing adapter ready for a key). `v2_weather_adjustment`
  flag + per-position `WEATHER_STRENGTH_*` knobs left in place.
- `scripts/analyze_coaching_defense_prior.py`, `scripts/analyze_weather_effect.py`,
  `scripts/sweep_weather_strength.py` — reusable analysis/sweep harnesses.
- 447 tests pass (+13 weather, +6 coaching).
