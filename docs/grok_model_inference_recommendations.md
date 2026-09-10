# Predictive model and inference recommendations

**Date:** 2026-08-23
**Branch:** `recommended_by_grok` (from `weekly-rankings-update0821` @ `b259507`)
**Scope:** Recommendations only. No application code, tests, or harness runs in this pass.
**Audience:** Anyone implementing the next weekly / draft / next-game projection work.

This document is a design and inference spec, not an audit of live Week 1 numbers. For the Week 1 correctness findings (injury map zip, stale prior-season injuries, vacancy double-counting, cache TTL, roster identity), see `docs/weekly_rankings_projection_audit_2026-08-20.md`. Those are prerequisites, not a scoring formula. This document assumes they will be fixed and describes what the models should become after that.

---

## Bottom line

The weekly model is a capable usage/shrinkage engine. `role_volume` is the only named component that clearly beat a trailing-4 baseline (MAE 4.710 → 4.422, rank-corr 0.654 → 0.689 on 8,107 paired player-weeks, 26/26 weeks). Extra matchup multipliers mostly measured as noise. Startable rank correlation barely moved.

The remaining error is inference, not missing clips:

- The product needs `P(A > B)` for start/sit. The model emits `E[points]`.
- Stats are generated independently, so the displayed line can be physically illegal (`rec > tgt`).
- Week 1 treats last year's missed games as a smaller current job.
- Defense ratings are a one-pass ratio with no sample-size shrinkage, then sliced into terciles that have 2–4 observations.
- Player Search still runs a weaker 60/40 twin and blends 70/30 with the market as two point estimates.
- The season simulator draws player-weeks independently, erasing game-script and injury correlation.

Do not start with XGBoost, and do not re-enable `volume_efficiency` or `game_env` as hard multipliers. They already lost the backtest for identifiable reasons. Fix the quantity being inferred, then re-measure with CRPS and pairwise start/sit accuracy.

---

## 1. Current architecture (as-is)

Three projection engines and one simulator, none of which share a posterior.

```
                    ┌─────────────────────────────────────────┐
  nflverse / PFF /  │  data/loaders.py                        │
  injury / odds     │  data/draft_sources.py                  │
                    └──────────────┬──────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌───────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Weekly Rankings   │  │ Player Search       │  │ Draft HQ            │
│ weekly_projections│  │ transforms.build_   │  │ draft_projections   │
│                   │  │ player_projection   │  │                     │
│ empirical-Bayes   │  │ 60/40 recent/season │  │ own rates × 17      │
│ rates, independent│  │ × matchup × pace    │  │ blended with ECR    │
│ stats, point mean │  │ then 70/30 market   │  │ rank curve          │
│ + one-sided linear│  │ blend per stat      │  │ season-total points │
│ calibration       │  └─────────────────────┘  └──────────┬──────────┘
└─────────┬─────────┘                                      │
          │                                                ▼
          │                                    ┌─────────────────────┐
          │                                    │ draft_season_sim    │
          │                                    │ independent gamma / │
          │                                    │ bootstrap draws     │
          │                                    └─────────────────────┘
          ▼
   Model Proj Pts  (point)
```

### What each engine infers today

| Engine | File | Inferential object | Output |
|---|---|---|---|
| Weekly Rankings | `data/weekly_projections.py` | Per-stat `E[rate]` with hardcoded `K`, times clipped multipliers | Point `Model Proj Pts` |
| Player Search next-game | `data/transforms.py` `build_player_projection` | 60/40 recent/season × opponent-allowed × pace | Point, then 70% market / 30% model |
| Draft HQ season | `data/draft_projections.py` | Own per-game rates blended with an ECR-indexed rank curve | Season-total point estimate |
| Draft season sim | `data/draft_season_sim.py` + `data/draft_weekly.py` | Weekly score draws, independent across players and weeks | Win distribution |

### What already works and must be kept

- Composition over a new stack: pace, injuries, scoring, and game-script bucket edges are reused primitives (`docs/weekly_projections_methodology.md`).
- Switchable `MODEL_FEATURES` and the paired A/B harness (`scripts/eval_weekly_model.py`). A component that cannot be turned off cannot be shown to help.
- Honest rejection: `volume_efficiency` (+0.051 MAE, 5/26 weeks) and `game_env` (+0.012 MAE) stay off, with the measurement written down.
- `role_volume` as share-when-active, not share-of-team-weeks-including-zeros. That distinction was settled by measurement (startable-RB MAE +0.28 when zeros were included).
- No target-week leakage in the main player rate (`week < as_of_week`).
- No season-long implied-points multiplier. `data/odds_market.py` already backtested that and it made season projections worse.

### What the last pass could not move

Quoted from the methodology, not re-litigated:

- Startable rank correlation is roughly flat (QB −0.021, WR −0.011, TE −0.053). The shipped components fix *level* (who is a starter, how big the top of a position should read), not which of two comparable starters outscores the other.
- `role_matchup` measured exactly neutral. A defense plays ~9 games; splitting those three ways leaves 2–4 observations; `ROLE_MATCHUP_K = 10` then shrinks the role rating back to the overall rating.
- Calibration is a monotone transform inside a position, so it cannot change ranks. On 2024–2025 it flipped startable-QB bias from +0.80 to −0.81.
- `teammate_vacancy` is unmeasured. The backtest runs with `apply_injury=False`.
- Wind was the largest measured environment effect and is unused because nflverse fills `wind` / `temp` after the game.

---

## 2. Target architecture

One generative weekly model. Everything else is a view or a prior on top of it.

```
                    ┌─────────────────────────────────────────┐
                    │  Evidence layer                         │
                    │  history, snaps, injuries (week-dated), │
                    │  depth chart, schedule, market lines,   │
                    │  FantasyPros weekly, forecast weather   │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  Hierarchical weekly engine             │
                    │  (single implementation)                │
                    │                                         │
                    │  1. P(plays) × E[share | plays]         │
                    │     × E[team plays]   → opportunity     │
                    │  2. efficiency ~ role/position mean     │
                    │  3. constrained stat line               │
                    │  4. player + defense ratings, shrunk    │
                    │  5. market / FP as likelihood           │
                    │  6. posterior predictive distribution   │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌───────────────┬────────┴────────┬───────────────┐
              ▼               ▼                 ▼               ▼
        Weekly board    Player Search     Draft HQ mean    Season sim
        mean, P(start),  same posterior    (17 × weekly    joint draws
        P(A>B), bands    no 60/40 twin     or season       given game
                                           analog)         environment
```

### Architectural rules

1. **One weekly engine.** `build_player_projection` becomes a single-player call into `build_weekly_projections` (or a shared core both call). Two different 60/40 vs shrinkage implementations of "next game" is a product bug, not a feature.
2. **The inferential object is a posterior predictive distribution**, not a calibrated point. The point total remains, as the mean of that distribution.
3. **Opportunity is generated first.** Yards and TDs are derived. The displayed line is a sample (or the mean) of a constrained joint, never four independent rates.
4. **Availability and role are separate latent states.** Last year's games played is not this week's snap share.
5. **The market is a likelihood, not a blend weight.** Coverage and residual variance set the weight. Rookies and Week 1 lean market; a week-10 veteran with 8 games leans history.
6. **Switchable features stay.** New components (`role_trend`, `redzone_tds`, hierarchical ratings, predictive distribution) go through `MODEL_FEATURES` and `scripts/eval_weekly_model.py` with a paired pool. The merge gate for ranking work is startable pairwise accuracy and CRPS, not whole-pool MAE.
7. **No future information in historical tests.** PFF route rate, pace, and any forecast weather used in a backtest must be as-of-week. The methodology already flags pace and PFF as leaks.

---

## 3. Architecture design changes

### 3.1 Split the weekly module into layers

`data/weekly_projections.py` is already the right home, but it currently mixes evidence construction, shrinkage, multipliers, vacancy, scoring, and Streamlit cache in one build function. Recommended shape, still inside this app's "composition not a new stack" rule:

| Layer | Responsibility | Stays / becomes |
|---|---|---|
| Evidence | as-of-week player rates, snap share, injuries with provenance, schedule, market lines | existing helpers, with week-dated injuries and current-roster pool every week |
| Role / availability | `P(plays)`, `E[share \| plays]`, `role_trend` state | new; `expected_snap_share` is the role half only |
| Opportunity | team pass/rush volume × player share | new; unimplemented `volume_faced` belongs here |
| Efficiency | hierarchical ratios shrunk toward role/position mean | retry of `volume_efficiency` with a different prior |
| Constraint | `rec ≤ tgt`, `td ≤ opportunities`, team shares sum | new reconciliation step |
| Matchup | iterated, shrunk player + defense ratings; continuous role | replaces one-pass ratio + terciles |
| Update | market / FantasyPros as noisy observations of the mean | replaces 70/30 and "comparison column only" |
| Predictive | gamma / Tweedie / NB draws; CRPS-able | new; reuse `draft_weekly` shape thinking, not its finish-rank key |
| Score | existing `score_projected_stats` on the mean line, and on draws | keep; apply to draws for the distribution of points |

Do not add a second package, a new web service, or an ML training repo. Keep numpy/pandas. Hierarchical ratings and variance-component `K` are closed-form or small iterative loops, same as the current vectorized game-script pass.

### 3.2 Unify Player Search with Weekly Rankings

`data/transforms.build_player_projection` is the old model: 60/40 trailing-4 / season average × `build_stat_allowed_matrix` × pace × alignment, then `build_market_projection` at 70/30.

That is a different answer to the same question the weekly board already answers, and it will disagree. Architecture change:

- Player Search "Next Game Projection" calls the weekly engine for that player-week.
- Market props update the posterior (section 4.5), they do not replace 70% of a weaker mean.
- Alignment (`build_alignment_multiplier`) remains a covariate on the receiving opportunity of WR/TE, not a third independent multiplier stacked on a different baseline.

### 3.3 Draft HQ: stop using the market as both prior and scorecard

`project_stat_lines` blends own history with a rank curve indexed by FantasyPros ECR, then the board is judged against ECR/ADP. That is circular for anyone whose rank is the thing being evaluated.

Recommended split:

- **Established players (evidence → 1):** generative usage model (multi-year recency-weighted rates, age, injury-history, FA adjustment — all of which already exist and measured well). ECR is not in the mean.
- **Rookies / role-changers / unsigned (evidence low):** ECR/ADP (and season-long books) as the prior, same Bayesian update as weekly. Role-change damping already tries to do this; make it the prior, not a slide toward a curve that *is* the market.
- **Season total:** either 17 × weekly posterior mean with bye/absence, or the current season analog run through the same opportunity-first / constrained line. Do not keep a third independent-stat generator.
- **VORP stays pure-model.** The methodology already forbids blending VORP. Keep that.

### 3.4 Season sim becomes a joint game model

`draft_season_sim.py` currently:

1. Draws each player-week independently from a finish-rank pool or a gamma with `FALLBACK_CV = 0.55`.
2. Knocks players out independently at a position absence rate, plus a known bye.

That overstates stacking value and understates shared script/weather/injury. Architecture:

1. Draw a **game environment** per team-week (plays, pass rate, margin, indoor, forecast wind).
2. Draw player outcomes **conditional on that environment**.
3. Availability = team-week factor + player factor, not 17 independent coins.
4. Key weekly pools on **projected/preseason rank or residual shape after scaling to the current mean**, not finish rank. Finish rank is selected on the same points being resampled.

The simulator is the only place the app already thinks in distributions. It should consume the weekly posterior, not a separate bootstrap keyed on a different ranking.

### 3.5 Evaluation architecture

Two scripts stay; their jobs stay distinct; the merge gate changes.

| Script | Job now | Job after |
|---|---|---|
| `scripts/validate_weekly_projections.py` | One honest pass vs trailing-4, MAE / rank-corr | Same, plus CRPS and startable pairwise accuracy. Still not a tuning knob. |
| `scripts/eval_weekly_model.py` | Paired A/B of `MODEL_FEATURES` | Same pairing rule. Default weeks include 1–4 with archetype slices. Injuries-on path once a week-dated feed exists. |
| `scripts/fit_weekly_calibration.py` | OLS `actual ~ a + b * projected` on 2021–2023 | Either isotonic/spline on residuals of the hierarchical mean, **per scoring format**, or deleted once the predictive distribution is calibrated (CRPS). |

Add a walk-forward runner that reports Weeks 1–4 separately, by archetype: returning starter, new starter, rookie, new team, backup, injury return. The methodology's headline evaluation of weeks 5–17 is the easiest window and the one the product is weakest outside of.

---

## 4. Inference changes

This is the core of the recommendation. Each subsection is a change to **what is being estimated**, not a new clip range.

### 4.1 Infer a distribution, not a number

Start/sit is `P(A > B)`, not `E[A] vs E[B]`. Two WRs both at 14.2 PPR are not interchangeable if one is 12 ± 3 and the other is 8 with a 25-point tail.

**Current:** `Model Proj Pts` is a shrunk mean, then one-sided linear calibration to fight overdispersion. Overdispersion is what you get when a noisy mean is treated as the prediction.

**Recommended:**

- Keep the current (or hierarchical) mean as the location of a predictive distribution.
- Points: gamma or Tweedie, dispersion estimated from *this* model's residuals, by position and projected volume. Do not reuse `FALLBACK_CV = 0.55` globally.
- Receptions / TDs: Poisson or negative binomial, consistent with the constrained line.
- Ship decision quantities: `P(WR12+)`, `P(bust < 8)`, `P(player A beats player B)`.
- Score with CRPS, log-loss, and pairwise start/sit accuracy. MAE can improve by shrinking everyone toward the mean while ranking gets worse.

This also makes `WEEKLY_CALIBRATION` (slopes 0.69–0.81, one table for all PPR formats) mostly unnecessary. If a location correction remains, fit it per scoring format, leave-future-years-out, and apply it to the mean *before* scoring draws — or rescale the displayed line so points match the stats (the audit's display inconsistency).

The season simulator and the cut Start%/Boom% columns in `data/draft_weekly.py` already pointed here. Recover the decision quantities on the weekly board, where a manager actually chooses between two players this week.

### 4.2 Opportunity first, efficiency toward the *role* mean, then constrain

Independent per-stat rates are why receptions can exceed targets. `volume_efficiency` diagnosed compounding (WR top 15%: targets +11%, receptions +16%, receiving yards +18%) and still lost, because efficiency was shrunk toward **the player's own prior-season efficiency**, which is also high for stars.

**Recommended generative order:**

1. Team pass attempts / rush attempts for the week (`volume_faced`: opponent and own pace split, not one `def_pace` number).
2. Player share of those attempts (`E[share | plays]` × `P(plays)`).
3. Efficiency ratios with empirical-Bayes shrinkage toward a **role/position** mean, evidence counted in opportunities:

```text
w      = n_i / (n_i + k)
theta  = w * own + (1 - w) * mu_role
```

`k` is estimated from variance components (James–Stein / reliability), by stat and position, on seasons outside the evaluation window. Stop hardcoding `STAT_K = {volume: 3, TDs: 5–6}` and `K_EFFECTIVE_RANGE = (0.7, 1.3)`.

4. Reconcile: `rec ≤ tgt`, `rec_td ≤ rec`, `rush_td ≤ carries`, yards consistent with opportunities. Team WR+TE+RB targets should not exceed team pass attempts. A Dirichlet-multinomial on team target shares, or a simple proportional repair after independent shares, is enough.

**Red-zone TDs** (`redzone_tds` in `MODEL_FEATURES`, unimplemented): TDs from red-zone carries/targets × a heavily shrunk TD rate. Raw TD per game should not be a first-class statistic. `build_redzone_usage` already exists from pbp.

### 4.3 Availability and role as two states

The cold-start path does:

```text
expected share  = player snap share across the whole prior team season
prior active    = average snap share in games he appeared in
role scale      = expected share / prior active share
prior rate     *= role scale
```

That is a valid backup detector and an invalid availability forecast. It is why Burrow / Daniels / Murray / Bowers / Lamb land as part-timers in Week 1.

`expected_snap_share` already learned that averaging zeros over team weeks wrecks returning starters. Leave that function as **role when active**. Add:

```text
E[volume] = P(plays) × E[share | plays] × E[team plays]
```

| State | Evidence | Must not use |
|---|---|---|
| `P(plays)` | current roster status, this week's injury designation (provenance-checked), start/depth-chart flag | last year's games played |
| `E[share \| plays]` | last N *appearances*, plus `role_trend` (EWMA / step-change in snap share) | whole-season share including absences |
| Week 1 prior | official depth chart, current eligible roster, conservative rookie/new-team prior | last season's IR games as a reduced job |

`role_trend` is declared and unimplemented. Implement it as a state: a 4% → 90% snap jump is a step, not a four-week average. Tyler Shough is the documented case this exists for.

Build the player pool from the **current eligible roster every week**, then left-join current stats and prior history. A returning IR or mid-week signing with zero current box-score rows should get an explicit low-confidence prior, not disappear once `cold_start` is false.

### 4.4 Hierarchical player and defense ratings, continuous role

`build_quality_adjusted_matchup` is a one-pass ratio of (what opponent did vs this defense) / (what that opponent does on average). The docstring already names the circularity: the baseline includes the game being rated, and neither side is adjusted for the other. Early samples are protected by `MATCHUP_CLIP = (0.75, 1.30)`, not by shrinkage toward 1.0.

`role_matchup` then splits those games into terciles. That cannot identify a "soft to possession, airtight deep" defense on ~9 games.

**Recommended:**

```text
y_ijt = alpha_player_i + beta_def_j + gamma_home + delta * role_i * beta_def_j + eps
```

- Iterate player and defense ratings to convergence (Massey / RAPM / mixed-effects). Partial-pool `beta_def` so a 1-game defense is almost league average. That is the actual replacement for clip ranges.
- Role is a **continuous covariate** (ADOT, target-share of touches), not `WR_SHORT / WR_MID / WR_DEEP`. A regularized `defense × ADOT` slope uses every game; terciles throw most of them away.
- Blend prior-season and current-season team ratings with `n / (n + K)`. Week 2 must not swap a full prior-year matrix for a one-game current matrix.
- Leave-one-out or simply iterate; do not include the target game in the rating used to price it (already true forward; make the historical adjustment the same).

Until defense ratings have sample-size shrinkage, adding more matchup slices cannot help. The 2024–2025 paired test already said so (`role_matchup` +0.000 MAE, −0.001 rank-corr).

### 4.5 Market as a likelihood

**Current:**

- Player Search: 70% sportsbook line / 30% model, per stat, two point estimates.
- Weekly Rankings: FantasyPros is a comparison column.
- Draft HQ: rank curve *is* ECR, then the board is scored against ECR.

**Recommended Bayesian update:**

```text
prior mean     = hierarchical weekly (or season) mean
likelihood 1   = player history, variance from residual model
likelihood 2   = sportsbook line, variance from book disagreement / vig / coverage
                 (MIN_COVERAGE and BOOK_WEIGHTS already exist)
posterior mean = precision-weighted combination
posterior var  = used for the predictive distribution
```

Weight slides automatically:

- Week 1, rookies, new team, thin coverage → market / FantasyPros dominate.
- Week 10, 8+ games, stable role → history dominates.

FantasyPros weekly projections are already imported (`data.draft_sources.fetch_fantasypros_weekly_projections`). Use them as the Week 1 prior the methodology itself says this app does not have, then let in-season evidence pull away.

Do not multiply a finished projection by implied team total. That was backtested and failed for season totals, and `game_env` failed as a hard weekly multiplier. Reintroduce implied total, venue, and **forecast** wind as regularized covariates *inside* the hierarchical mean, with coefficients estimated on 2019–2023 (already measured: QB elasticity 0.416, wind 0.880 at 15+ mph outdoors). A forecast feed (NWS / Open-Meteo at lock) is required; nflverse `wind` after kickoff is forbidden in backtests.

### 4.6 Estimate shrinkage from data

Almost every constant is a reasoned guess that was then A/B'd as a unit. Next step is to **fit hyperparameters out of sample, then freeze them**.

| Quantity | Fit on | Method |
|---|---|---|
| Volume vs TD `K` | week-to-week reliability, 2019–2023 | variance components |
| Recency decay | same, as a lag kernel | MLE on predictive log-likelihood (replace `RECENCY_DECAY = 0.85`) |
| Defense rating `K` | games faced | hierarchical SD |
| Role-confidence effect on `K` | reliability by snap-share band | replace `K_EFFECTIVE_RANGE` |
| Residual dispersion | this model's residuals, by position and projected volume | feeds the predictive distribution |
| Vacancy absorb / growth | reconstructed week-dated inactives | until then, vacancy is editorial, not inference |

Do not tune these against 2024–2025. The methodology already forbids that.

### 4.7 Joint season-sim inference

Draw environment first, then players. Correlate teammates through the game. Cluster availability. Key pools on projected rank or residual shape. Use the weekly posterior's dispersion rather than a position-typical CV.

---

## 5. Recommended coding changes

No code in this branch. The list is the intended surface area so a later implementation pass does not wander.

### 5.1 Weekly engine — `data/weekly_projections.py`

- Implement `role_trend`, `redzone_tds`, `volume_faced` as real, switchable components, or remove them from `MODEL_FEATURES` so the evaluator cannot report a no-op as a variant.
- Add `availability` (or split `role_volume` into role vs `P(plays)`). Cold-start must not scale prior rates by whole-season snap share.
- Rebuild `volume_efficiency` with role/position-mean shrinkage and a constraint pass; keep it off until the paired harness says otherwise.
- Replace tercile `build_player_roles` / `build_role_matchup` with continuous role covariates and iterated, shrunk ratings. Keep `role_matchup` as a switch until the replacement wins on startable pairwise accuracy.
- Add a predictive-distribution object on the result frame (mean, sd or gamma params, optional quantiles). Do not push calibration into individual stats; if the mean is corrected, rescale the displayed line or show the correction.
- `_cold_start_pool`: filter to eligible roster status; use player IDs; map FB/RB as one role family (Juszczyk case).
- In-season pool: current eligible roster left-joined to stats, not `_season_totals(hist)` only.
- `_injury_multipliers`: pair player and status before `dict(zip)`; refuse a feed whose `attrs['season']` is not the projected season; preserve exact status (stop collapsing everything `< 0.9` to `Out/Doubtful`).
- `redistribute_vacated_usage`: redistribute only the lost portion (`full − discounted`), not 75% of full volume on top of a 40% doubtful remainder.
- Blend prior and current defense/pace with evidence weight. Neutral opponent fallback if `load_schedule` is empty — do not drop the board.
- Recency: name and test calendar-week vs games-played decay. A bye currently decays as if the player played.

### 5.2 Shared next-game path — `data/transforms.py`

- `build_player_projection`: delegate to the weekly engine. Leave alignment as an input to that engine, not a parallel formula.
- `build_market_projection`: return line + implied variance / coverage, not a replacement mean. The update lives in the weekly engine.
- Keep `score_projected_stats` in lockstep with `apply_scoring_and_percentiles` (already a documented twin). One function should score both the mean line and each draw.

### 5.3 Draft — `data/draft_projections.py`, `data/odds_projections.py`

- Established-player mean does not include the ECR curve. Rookies/role-changers take ECR/books as prior (section 3.3).
- Season line uses the same opportunity-first + constraint path as weekly.
- `blend_market_into_projection`: precision-weighted update, coverage-aware, same as weekly. `MIN_COVERAGE` stays.
- `MEDIAN_TO_MEAN` remains a small explicit correction until the predictive distribution makes it redundant.

### 5.4 Simulator — `data/draft_season_sim.py`, `data/draft_weekly.py`

- Joint game-environment draws (section 3.4).
- Weekly pools keyed on projected rank or residual shape, not finish rank (`RANK_WINDOW` around the wrong index).
- Consume weekly posterior dispersion when a player has one.

### 5.5 Evaluation — `scripts/eval_weekly_model.py`, `scripts/validate_weekly_projections.py`, `scripts/fit_weekly_calibration.py`

- Metrics: CRPS, pairwise start/sit inside `STARTABLE_N`, archetype slices, weeks 1–4.
- Format-specific calibration files, or disable calibration outside Full PPR until those exist.
- Strict as-of-week for PFF and pace in historical runs; omit PFF from backtests if it cannot be dated.
- Injuries-on evaluation only after a week-dated injury reconstruction exists. Until then, vacancy stays flagged unmeasured.

### 5.6 Identity, pool, and freshness (model-adjacent; blocking for inference)

These are in the 2026-08-20 audit. They are listed here because the inference in §4 is wrong if they stay:

- Preserve raw game `team` separately from current roster team (`data/loaders.py` overwrites historical team).
- Player ID joins, not name + exact position.
- Cache keys include data version / mtime / TTL (`load_year_data`, `load_team_pace`, `load_schedule`, `build_weekly_projections`). Visible refresh.
- Do not apply prior-year injury fallbacks to the projected season.

### 5.7 Presentation (only what inference requires)

Weekly Rankings should show, at least in weeks 1–4: cold-start flag, current/prior blend weight, `P(plays)`, role share, matchup rating sample size, market weight, calibration/mean correction, and a floor/ceiling or 25th/75th from the predictive distribution. Without that, the board cannot be audited and the new objects will rot.

Do not imply the displayed stat line sums to `Model Proj Pts` unless they are forced to match.

---

## 6. Other changes (in scope of "make the models better")

### 6.1 Forecast weather

Largest measured unused effect in the game-environment study. Add a lock-time forecast source. Never train or backtest on nflverse post-game `wind` / `temp`.

### 6.2 Week-dated injuries

The live path wants "most recent designation." The backtest and vacancy measurement want designation as of `week`. Store or reconstruct weekly injury rows. Without this, `teammate_vacancy` cannot be falsified and must stay conservative / switchable.

### 6.3 Depth chart as Week 1 opportunity prior

New play-callers, free-agent moves, promotions, and rookies are not in last year's rates. Official depth chart / start designation should move **opportunity**, not apply a points bump. Malik Willis vs a new starter is a role-prior problem, not a shrinkage-K problem.

### 6.4 Multi-year prior

The prior is only `year - 1` totals / games. A one-game cameo is trusted as a complete prior; a veteran who missed all last year falls to a position baseline. Keep a games-weighted multi-year prior with age/role decay. Distinguish no evidence from thin evidence. Draft HQ already has `SEASON_RECENCY_WEIGHTS` and `SEASON_COMPLETENESS_FLOOR`; weekly should share that idea.

### 6.5 Do not add a black-box learner as the engine

n ≈ 8,000 noisy player-weeks, high game variance, inspectable stat line is the product. If a learner is used at all: residual model on top of the hierarchical mean, walk-forward, features known at lock (role, opponent rating, implied total, forecast wind, rest, vacancy). It must emit a legal stat line. A MAE win that cannot explain carries vs targets is a worse product.

### 6.6 Documentation drift to fix when code exists

- Methodology still describes the old `0.6 × trailing-4 + 0.4 × season-to-date` current rate in one section; code uses recency/matchup/rematch weights.
- `RECENCY_DECAY` is described as per game and implemented as calendar-week distance.
- `MODEL_FEATURES` lists unimplemented names the evaluator treats as real variants.
- Player Search and Weekly Rankings should share one methodology section once they share one engine.

---

## 7. What not to do

These are recorded so they are not re-proposed from intuition. All are already measured or structurally circular.

| Proposal | Why not |
|---|---|
| Re-enable `volume_efficiency` as currently written | Shrinks stars toward their own high prior efficiency; +0.051 MAE, 5/26 weeks |
| Re-enable `game_env` as a hard multiplier | +0.012 MAE at measured elasticity; put the same signals inside a regularized mean instead |
| Multiply season projections by Vegas implied points | 748 player-seasons, worse at every strength (`data/odds_market.py`) |
| Workload penalty for 300-touch backs | Measured backwards: heavy usage predicted *more* games and *better* retention (`docs/draft_hq_methodology.md` §2.5d) |
| Young-player age boost | Evidence already low; consensus curve carries them; MAE slightly worse |
| Steeper age band past 34 | Neutral on every metric; n is tiny |
| QB/WR injury-history markdown | QB MAE got worse (job loss, not health); WR borderline |
| Fit `STAT_K` / calibration on 2024–2025 | The evaluation window; the methodology forbids it |
| Wire nflverse `wind` into the live model | Filled after the game; backtest would lie |
| Replace the engine with XGBoost on box scores | Wrong object, leak-prone, loses the stat line |
| Average snap share including missed weeks as the role | Returning-starter startable-RB disaster, already measured |
| Blend VORP toward ADP | Explicitly forbidden; VORP stays the model |

---

## 8. Key decisions

1. **Posterior predictive, not calibrated point.** Start/sit and ranking among starters are probability questions. MAE on the full pool is a secondary diagnostic.
2. **One weekly engine.** Player Search delegates. No 60/40 twin.
3. **Opportunity → efficiency → constraint.** Independent rates are illegal football and were the reason `volume_efficiency` was even built.
4. **Efficiency prior is the role/position mean, not own last year.** That is the actual retry condition for `volume_efficiency`.
5. **Availability ⊥ role.** `expected_snap_share` stays share-when-active. `P(plays)` is a new state.
6. **Iterated, shrunk ratings + continuous role.** Tercile matchups cannot identify with 2–4 games. Clip ranges are not shrinkage.
7. **Market is a likelihood.** FantasyPros weekly is the Week 1 prior. Coverage sets weight. 70/30 is retired.
8. **Hyperparameters fit outside 2024–2025, then frozen.** Components still A/B'd on the paired harness.
9. **No black-box replacement of the mean model.** Residual learner only, and only if it emits a legal line and wins CRPS / pairwise, not just MAE.
10. **Vacancy stays switchable and conservative until week-dated injuries exist.** Unmeasured live adjustments do not get aggressive constants.
11. **Season sim is joint.** Independent player-week draws are not a win model.
12. **Correctness blockers from the 2026-08-20 audit ship before or with the first inference PR.** A better likelihood on the wrong player or a stale injury is not better inference.

---

## 9. Suggested implementation order (not done here)

Each step is independently measurable. Do not skip evaluation architecture; otherwise the next pass will keep shipping level fixes that leave startable rank-corr flat.

1. **Evaluation gate:** CRPS (once a distribution exists, even a naive gamma around the current mean), pairwise start/sit, weeks 1–4 archetype slices, as-of-week PFF/pace discipline. Wire this before claiming wins.
2. **Correctness blockers** from the audit: injury zip/provenance, vacancy conservation, roster eligibility, player IDs, cache freshness, historical team preserved.
3. **Availability vs role + `role_trend`.** Fixes the structural Week 1 failure.
4. **Opportunity-first line, role-mean efficiency, physical constraints.** Retry of `volume_efficiency` under the prior that actually regresses stars. Implement `redzone_tds` and `volume_faced` here.
5. **Iterated player/defense ratings; continuous role.** Replacement for tercile `role_matchup`.
6. **Predictive distribution on the weekly board** (mean, bands, `P(A>B)`). Recalibration of points as the mean of draws.
7. **Market / FantasyPros as Bayesian update.** Delete 70/30. Player Search calls the weekly engine.
8. **Forecast wind + regularized environment covariates** (not hard multipliers).
9. **Draft HQ mean uses the same generative line;** ECR only as prior for low-evidence players.
10. **Joint season sim** consuming the weekly posterior.

---

## 10. Open questions (product, not math)

These are not resolved in this document. They do not block the architecture.

1. **How much of the predictive distribution to show on Weekly Rankings?** Mean-only keeps the current table; adding floor/ceiling/`P(start)` is the point of §4.1. Recommendation: mean + 25th/75th + pairwise vs the other player on the row when a comparison is open.
2. **FantasyPros weekly as a required Week 1 prior, or optional when uploaded?** The methodology already treats FP as the better opener. Recommendation: use it when present; cold-start role/availability prior when not; never silently mix without a source label.
3. **Lock-time weather vendor.** NWS vs Open-Meteo vs a paid feed. Any is fine if the backtest uses *that* forecast as of lock, not the game's realized wind.
4. **Whether a residual GBDT is ever in scope.** Default is no. Revisit only after steps 1–7, and only on residuals.

---

## 11. Files this design would touch (later)

Documentation-only on this branch. Future implementation surface:

- `data/weekly_projections.py` — engine
- `data/transforms.py` — `build_player_projection` / scoring twin
- `data/loaders.py` — team identity, pace as-of-week, cache keys
- `data/draft_sources.py` — week-dated injuries, FantasyPros weekly as prior
- `data/draft_projections.py` — season generative line, ECR as prior only
- `data/odds_projections.py` — precision-weighted market update
- `data/draft_season_sim.py`, `data/draft_weekly.py` — joint sim
- `scripts/eval_weekly_model.py`, `scripts/validate_weekly_projections.py`, `scripts/fit_weekly_calibration.py`
- `ui/tabs/rankings.py`, `ui/tabs/draft_hq.py`, Player Search snapshot — display of distribution and blend weight
- `docs/weekly_projections_methodology.md` — replace the stale 60/40 description when the engine changes
- `tests/test_weekly_projections.py` (and siblings) — constraint tests, availability vs role, paired injury maps

No tests were added or run for this branch.

---

## 12. Relationship to existing docs

| Doc | Role vs this one |
|---|---|
| `docs/weekly_projections_methodology.md` | How the shipped weekly model is made, and the measured accept/reject of each component. Still the source of numbers cited here. |
| `docs/weekly_rankings_projection_audit_2026-08-20.md` | Live Week 1 correctness. Do those first. This document does not repeat the player-by-player ECR table. |
| `docs/draft_hq_methodology.md` | How the shipped draft engine is made, including age/injury/FA measurements this design keeps. |
| `HANDOFF.md` | Engineering history, gotchas, cache rules. |

This document is the proposed *next* architecture for inference. It does not claim the current board is unused or that `role_volume` should be ripped out. It claims the next gains will not come from another clipped multiplier.
