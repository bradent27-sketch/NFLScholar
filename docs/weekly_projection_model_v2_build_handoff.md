# NFL Scholar — Weekly Projection Model V2 Build Handoff

**Status:** partial implementation on `feat/weekly-projections-v2-foundation`; V2 remains experimental and is not the released default. The QB starter gate and partial-game data-integrity safeguards apply to both model versions.  
**Scope:** the Weekly Fantasy → Weekly Rankings projection engine only.  
**Primary implementation model:** Codex 5.3 through OpenCode, using the user's already-configured API key. Do not request, print, store, or commit that key.

This is the execution contract for the next major projection-model pass. It turns the
product decisions into an ordered build plan a fresh coding agent can follow without
reopening settled design questions. Read it before changing
`data/weekly_projections.py`, `ui/tabs/rankings.py`, or the weekly-model tests.

## Read order and working rules

1. Read this file in full.
2. Read `HANDOFF.md`, especially its Weekly Rankings / weekly-projection notes and its
   regression-testing rules.
3. Read `docs/weekly_projections_methodology.md`; it describes V1 and contains useful
   historical measurements, but V2 decisions in this file take precedence where they
   conflict.
4. Inspect the existing code and tests before proposing an implementation phase.

All V2 work must happen on a **new feature branch** (for example
`feat/weekly-projections-v2-foundation`). Keep commits narrowly scoped. Do not push,
open a PR, or merge into `main` until the user has received a plain-language change
summary and explicitly approved the next GitHub action.

Keep `app.py` thin and preserve lazy tab execution. Do not add a heavy dependency just
for one calculation. Use the existing data-loader patterns and retain the application's
data/privacy guardrails. Never treat a source with only season totals as if it were
available before a historical target week.

## Product objective

Produce auditable weekly fantasy projections for **QB, RB, WR, and TE** that combine
current role, relevant prior evidence, opponent vulnerability, availability, and game
context without future leakage. The projection must be useful for start/sit and player
prop research: a user should be able to select a player in the rankings table and see
what drove the number, not merely receive a black-box point total.

The model should improve only when an addition beats a declared baseline on honest
out-of-sample tests. A feature that is football-plausible but unmeasured, neutral, or
harmful may remain experimental; it must not quietly become an unconditional default.

## Current implementation snapshot (not a claim of accuracy improvement)

The first V2 foundation pass implemented the following behind the **V2 (experimental)**
model selector in Weekly Rankings. It did not change the released V1 default.

- A named V2 feature set, raw/calibrated point fields, source-cutoff metadata, and a
  backend explanation payload. The player-row dialog renders that payload; it never
  recomputes the model in the UI.
- A historical guard: PFF season-total receiving data, live injury status, and live
  market script inputs are excluded when the loaded season already contains the target
  week. Historical pace uses a weekly box-score pass-attempt-plus-rush-attempt proxy
  instead of the future-aware full-season pace loader.
- Current defense profiles blend into prior-year quality-adjusted profiles by defense
  game count (four current games is a 50/50 bridge). Current player-history rates
  remain opponent-adjusted.
- Continuous target-earner/rushing/receiving/ADOT role profiles, plus smooth blends
  across the legacy nearby-role defensive tables. This removes a player-side hard role
  cliff; it does **not** yet create true alignment-specific defense data.
- Confirmed WR/RB opportunity changes can reduce volume shrinkage, while TD rates stay
  conservative. Comparable-role TD rates can include a modest second prior season.
- Explicit QB-passing, QB-rushing, RB-rushing, and RB-receiving trace channels. QB and
  RB rushing remain position-separated in both the player and defensive inputs.
- Availability probability separated from workload-if-active, and a role-specific live
  vacancy allocator: QB passing stays with QB candidates; RB carries/targets split;
  WR/TE targets do not automatically spill to RBs; unallocatable volume is ledgered.
- A 43-test offline projection suite, including an end-to-end mocked V2 run that proves
  a future-week perturbation cannot change an earlier as-of projection.

### Follow-up implementation: QB1 selection, partial-game screening, robust all-stat defense profiles, and full trace

The following additions were made after a live Week 1 audit exposed real
under-projected passing volume for expected starting quarterbacks who missed
substantial time in the prior year. They are implemented in both model
versions where applicable, but have **not** yet been claimed as a measured
accuracy improvement.

- **Expected QB1 policy and nonstarter gate:** the weekly model does **not** consume
  the generated Depth Charts table as a starter source. `data/qb1_overrides.csv` is
  the explicit, user-controlled selection layer for any upcoming week. At a cold
  start, a clear prior-season incumbent (at least 65% of team weeks) can be selected
  automatically; otherwise one manual choice is required. In season, a QB is
  automatic only when he was recently active for that team and his most recent
  eligible game shows a clear full-snap starter role. An old starter cannot win on
  September snaps after another QB has taken the recent starts. Exactly one selected
  QB receives normal QB volume; all nonstarters are explicitly zeroed rather than
  retaining a relief-game per-appearance projection. An unresolved room has no
  projected QB volume until the user selects one, which is auditable in both the
  rankings dialog and the Depth Charts selector. The initial 2026 file is a
  reviewable manual seed (including the user's Dart/Watson choices), not a dynamic
  ECR input to the model.
- **Returning WR/RB/TE role restoration:** at a true cold start, a player who
  remains on the same team can recover his per-active-game snap role rather than
  being treated as inactive for every game missed last year. This is gated to
  meaningful prior evidence (RB: 8 games; WR/TE: 6), same-team continuity, and a
  continuous active/pre-absence role signal with a 95% cap. It intentionally excludes
  rookies, sparse backups, and new-team players; it is a preseason role policy, not an
  injury forecast.
- **Partial-game player-history screen:** historical box scores have no trustworthy
  game-level injury timestamp, so the model does not infer an injury from a quiet
  fantasy line. It excludes an individual game from that player's full-game rate,
  current-evidence count, role trend, and snap expectation only when recorded snaps
  prove a narrow interruption: a QB split/relief game, an abrupt <=50% snap drop
  after an established role, a paired low-history replacement after that teammate's
  exit, or sharply reduced work in a 28+ point winning blowout. Missing/unmatched
  snap data is never classified. Raw team-game data remains intact for defense
  profiles, because the defense still faced the complete offense that day. Trace
  output exposes excluded counts and reasons; this is a data-integrity safeguard,
  not an injury forecast or a measured accuracy claim.
- **Prior-season defense evidence:** an offseason prior-defense profile now
  uses a 75% full-season baseline plus a 25% late-season tilt. It replaces
  the old use of the normal 0.85 per-game decay across the season, which
  could make one late defensive game many times more influential than an
  early game in a Week 1 matchup. In V2, the resulting prior profile still
  transitions toward current-season evidence by defense-game count.
- **All-stat defense estimator:** every projected stat now uses an
  offense-position team-game profile. The model sums all players at the same
  position into one offense-versus-defense game, normalizes that position total
  against the offense's own baseline, and computes a recency-weighted pooled
  observed/expected factor with four neutral league-average games of shrinkage.
  It replaces the fragile player-game ÷ player-season-average calculation, so a
  spot starter, injury fill-in, or low-volume player cannot dominate a defense
  rating through a tiny denominator. The broad profile includes real zero-output
  positional games when the weekly file has no stat row. QB rushing, RB rushing,
  RB receiving, WR receiving, and TE receiving remain separate channels; QB
  rushing yards floor historical team totals at zero to prevent kneel-downs from
  creating negative denominators.
- **Historical offense identity:** raw weekly `game_team` and `game_opponent`
  are preserved before a current-roster merge. All historic defense team-game
  calculations use `game_team`, never the player's later roster team after a
  trade. `game_id` reconstruction is only a compatibility fallback for old cached
  frames.
- **Role/efficiency safety:** role-conditioned tables now aggregate at the same
  team-game grain and keep their evidence shrinkage. QB passing bypasses a role
  overlay in both the direct-rate and optional volume×efficiency paths, so an
  optional feature cannot reintroduce the old player-row sensitivity.
- **Projection decomposition:** each stat trace now records raw and
  role-scaled prior rates, blend evidence/weight, defense source and factor,
  script, pace, availability, environment, optional efficiency rebuild, and
  pre-/post-vacancy values. The Weekly Rankings dialog keeps a compact
  summary table and offers a per-stat input/evidence expander. Its stat line
  is refreshed after vacancy redistribution so it cannot disagree with the
  selected ranking row. It also identifies the expected QB1/nonstarter-volume status,
  partial-game exclusions, returning-role restoration, and the defense
  estimator/role-overlay decision.
- **Regression coverage:** the offline suite now has 64 tests, including manual
  QB1 selection, automatic incumbent selection, every projected stat's
  spot-starter partition invariance, role-overlay invariance, QB kneel safety,
  historical-trade identity, QB efficiency safety, prior-defense recency shape,
  clear partial-game/replacement/blowout cases, nonstarter-QB suppression, and
  trace-contract checks.

### Follow-up implementation: V2 preseason RB allocation and PFF alignment archive

The following remains experimental and V2-only. It is a data-contract and role-correction
pass, not a claim of an out-of-sample fantasy-point improvement.

- **Identity and literal depth order:** cross-season player joins now prefer stable GSIS/PFF
  identifiers, with a small reviewed alias table only for ID-less source spelling changes
  (for example Kenny/Kenneth Gainwell). Historical `game_team` remains immutable after a
  trade. Ourlads' literal RB order is preserved: an unmatched RB2 cannot silently promote
  an RB3 to rank 2.
- **Functional fullbacks:** roster `depth_chart_position`, then Ourlads, then historical
  source position determines FB versus core RB. A fullback is kept as a transparent,
  low-usage RB-display row from his own historical touch rate, but is excluded from RB
  capacity, population fallback, and vacancy pools. A current functional-position map
  persists this distinction after the first played week.
- **Team-constrained preseason RB allocator:** for live Week-1-style V2 runs, core-RB
  snaps, carries, and targets are separate finite team capacities. Only credible active
  RBs receive allocation; the remaining capacity is an explicit other/unallocated bucket.
  Score inputs are active/whole/pre-absence role, continuity, literal Ourlads rank, draft
  signal, availability, and teammate competition. An uncharted team cannot convert the
  cold-pool median placeholder into a reserve role; it needs observed prior evidence or
  meaningful draft capital.
- **Vacancy correctness:** RB carries and targets use separate role-aware recipient shares;
  WR/TE targets stay inside the healthy pass-catcher pool, and FB/WR absences cannot boost
  a lead RB's carries. Prior-season injury provenance is rejected for a new Week 1 run.
  The popup ledger records capacity, recipient, source functional role, provenance, and
  explicit unallocated volume.
- **Manual weekly PFF alignment archive:** `data.pff_alignment` reads a time-safe pair of
  league-wide reports per week (`receiving_summary` plus `receiving_concept`) and an
  optional provenance manifest. Player slot/non-slot profiles are shown in the popup.
  Defense evidence comes only from weekly **offensive** PFF data mapped to the scheduled
  opponent and aggregated at a team-game grain; the incompatible defender `slot_coverage`
  report is never used. WR/TE target/reception/yard residuals were a bounded audit preview
  only through 2026-08-23; on 2026-08-24 the residual was wired into scoring behind the
  `v2_pff_alignment_matchup` feature flag and run through the locked out-of-sample backtest
  this note used to require — see `docs/weekly_projections_methodology.md`'s
  `v2_pff_alignment_matchup` section for the numbers. **Result: rejected**, same bar as
  `volume_efficiency`/`game_env` — it loses MAE and rank-corr on the startable WR/TE pool
  that actually drives a lineup decision, worst on TE. The mechanism, code, and lookup
  helpers all stay in place (reachable by explicit feature name, excluded from
  `DEFAULT_FEATURES`); TD, RB/FB, and missing data remain neutral regardless. **Update, same
  day:** re-enabled in `V2_EXPERIMENTAL_FEATURES` only, at the user's request, purely so the
  active per-player residual is visible on a real V2 board while looking for a fixable
  cause — see the methodology doc's own update note. `DEFAULT_FEATURES` is unaffected.

### Still deferred because the current sources cannot support an honest version

- **WR slot/wide and TE slot/inline matchup math:** the dated weekly feed this used to be
  blocked on now exists for 2025 (`pff_imports/2025/weekly/`) and was tried — see the
  "Follow-up implementation" section above: built, backtested, and rejected 2026-08-24, not
  for a data-availability reason but because 2025 is still the only season with weekly-grain
  files, so the defense-side slot/non-slot split is thin (a handful of games per team) even
  with the existing shrinkage. Revisit once more weekly-grain seasons accumulate, not by
  retuning the shrinkage constants against this same single season.
- **Depth × field location and QB pressure/sack inputs:** play-by-play can support a
  future depth/location study, but the current pass has not validated an incremental
  projection benefit and does not fabricate pressure data from a season summary.
- **As-of roster/depth and historical injury snapshots:** current roster/name fallbacks
  are retained, but there is no date-correct depth-chart or injury-report archive for a
  leakage-safe historical vacancy/back-up evaluation. Live V2 degrades gracefully and
  records unallocated volume when a credible replacement is absent from the pool.
- **Team passing-script, weather, recalibration, uncertainty intervals, and locked
  multi-season release evaluation:** these require the focused, predeclared studies in
  components 11, 14, and 16. The old calibration remains visible/toggleable; it has not
  been re-fit or declared a V2 improvement.

## Decisions already made

- Preserve V1 while V2 is developed. Every meaningful V2 component needs a named
  feature switch so it can be compared alone, in combinations, and rolled back.
- Never use an after-the-fact value for a historical `as_of_week` projection. This is
  especially important for roster state, injuries, PFF season summaries, team pace,
  market lines, and game results.
- Keep **raw model projection**, **calibrated model projection**, and **market
  projection** separate. Market lines are display/comparison evidence, not a silent
  input to the model unless the user later explicitly authorizes a market-blended model.
- Player roles are continuous/mixed profiles, not one hard label. A player who is 65%
  slot and 35% wide must retain both pieces of information.
- Defensive weakness is measured from what a defense allowed to all relevant plays and
  players in a role, adjusted for the offensive quality and opponent context it faced;
  it is not just a list of famous players it happened to allow yards to.
- QB rushing and RB rushing must remain separate player-side and defense-side channels.
  QB goal-line rushing belongs in QB rushing/TD evidence, never in RB rushing evidence.
- Current WR and RB **volume** may adapt quickly after a confirmed role change. TDs
  remain much more conservative and can use two prior seasons when prior roles are
  sufficiently comparable.
- Early-season defense must begin partly tied to prior-year defensive evidence, but
  adapt materially faster than player priors because coaching, personnel, and scheme
  can change in an offseason.
- A pre-kickoff questionable designation is not proof that a player will receive a
  reduced workload if active. Availability probability and conditional workload are
  separate concepts.
- The ranking-table drilldown is a spacious popup/dialog, not extra columns that make
  the weekly table unreadable.

## V2 components

### 1. V2 shell, experiment registry, and output contract

**Build**

- Keep the V1 output path intact.
- Add a clearly named V2 configuration/feature registry. Components must be independently
  enabled for tests and grouped only after individual evaluation.
- Define a stable projection result contract containing: player identity, position,
  opponent, target week, projected stat line, raw fantasy points, calibrated fantasy
  points, active feature flags, and a structured explanation payload.

**Acceptance criteria**

- Existing weekly rankings render with V2 switches off.
- A V2 run can be reproduced from the same year/week/as-of-week/configuration.
- Tests can select any component combination without UI state.

### 2. Canonical identity, roster, and as-of-week data contract

**Build**

- Establish a canonical player/team identity layer for game stats, play-by-play, PFF,
  roster/depth information, injuries, and odds. Prefer stable IDs; document every
  name/team fallback and its confidence.
- Add an explicit data cutoff contract to each relevant loader. A historic Week N
  projection may consume only information public before that week's kickoff.
- Add an as-of roster/depth snapshot policy. Do not use the newest current roster to
  infer a starter or backup for a past week.
- Record missingness and source timestamp/season coverage in the explanation/debug
  payload rather than turning unknown data into a confident zero.

**Known hazards to eliminate**

- Current PFF season-total files lack a week/date cutoff and therefore cannot be used
  directly in an honest historical weekly backtest.
- The current full-season pace helper leaks future games when called for a historical
  week.
- The current injury source is suitable for live use but not historical backtesting
  without a date-correct report snapshot.

**Acceptance criteria**

- Deliberately perturbing a future game cannot alter an earlier projection.
- Every historical test explicitly reports its cutoff and skips/degrades gracefully for
  unavailable data.

### 3. Baselines, priors, and adaptive player blending

**Build**

- Retain stat-level rather than one-size-fits-all shrinkage. Volume, efficiency, and TD
  rates stabilize differently.
- Use current-season evidence adjusted for opponent quality, blended with prior player
  evidence, then position baseline only when player evidence is insufficient.
- For players with defined, comparable roles, allow a second prior season for TD-rate
  evidence. Weight older evidence less and exclude it when the role/team context is not
  comparable.
- Implement an explicit, tested role-change detector for WR/RB volume. It may reduce
  the effective current-season shrinkage only when snaps/routes/opportunity evidence
  confirms a durable role change.

**Target behavior**

If a receiver averaged 6 targets in the comparable prior role and opens the new season
with three genuine 9-target games plus elevated route participation, the volume estimate
should land near **8.0–8.25 targets**, not the V1-style 7.5. Do not grant that aggressive
weight after three anomalous games without role confirmation.

**Acceptance criteria**

- Unit tests cover 1-game, 3-game, confirmed-role-change, and no-role-change examples.
- TD changes do not inherit the aggressive WR/RB volume settings.
- A trace lists current evidence, prior-season evidence, role-comparability decision,
  effective sample size, and final blend weights.

### 4. Defensive evidence, prior-year bridge, and schedule adjustment

**Build**

- Replace a defense-only-season view with adjusted defensive evidence: compare what the
  defense allowed against the expected output of the offensive players/units it faced.
- Maintain separately estimated defensive profiles for the channels in components 6–10.
- Blend early current-season defensive evidence with prior-year evidence:

  `defense_profile = alpha * current_adjusted_profile + (1 - alpha) * prior_profile`

  where `alpha` grows with effective sample size. Tune the prior strength by channel and
  validate it; allow faster adaptation than player priors.
- Make schedule/opponent-quality adjustments component-level. A defense should not look
  bad merely because it faced several elite offenses, nor good merely because it faced
  weak ones.

**Acceptance criteria**

- Week 1 has a documented prior-year fallback.
- Weeks 2–5 visibly transition toward current-season evidence.
- Tests show a strong opponent schedule does not mechanically inflate a defense's
  apparent softness.

### 5. Role and alignment profile layer

**Build**

- Create a reusable player-role profile with confidence/sample fields, not hard tercile
  buckets.
- WR: slot rate, wide/non-slot rate, route participation, target share, target-earner
  rank within offense, ADOT, and short/intermediate/deep usage when source support exists.
- TE: slot rate and inline/non-slot rate as first-class fields. If credible TE-wide data
  becomes available, add it separately; until then, disclose that non-slot is primarily
  inline and may include wide outliers.
- RB: carries, targets, receiving route participation, rushing share, and receiving-back
  role. Do not collapse rushing and receiving into a single RB bucket.
- QB: passing style remains continuous (for example ADOT/completion profile) rather than
  a hard passer label. QB rushing is a separate profile.

Use existing PFF alignment data only where its time availability is honest for the run.
When historical, as-of-week alignment is unavailable, use an eligible play-by-play
derivation or omit the feature rather than leaking a season total.

**Acceptance criteria**

- One player's profile can contain mixed alignment weights that sum sensibly.
- Low-sample profiles shrink toward an appropriate position/team baseline and expose
  their confidence.

### 6. QB projection: passing volume, efficiency, and TDs

**Build**

- Project passing attempts, completions/efficiency, passing yards, and passing TDs as
  related but distinct channels.
- Keep ADOT as an input, and estimate matchup similarity through continuous/nearby
  defensive profiles rather than assigning a QB to a single hard bucket.
- Test whether completion rate, pressure/sack tendency, and depth × location add distinct
  predictive signal. Do not add correlated features merely because they sound useful.
- Plan a separate team-level passing-script study using team scoring and margin. Test
  effects on attempts, passing yards, and passing TDs; do not simply apply the old
  skill-player game-script curve to all QB passing stats.

**Explicit non-goal for this phase**

Designed-rush versus scramble differentiation is not required now. The model needs
separate QB rushing evidence, not that finer split yet.

**Acceptance criteria**

- Passing volume/yardage/TD estimates each cite their contributing inputs in a trace.
- Any team-script multiplier ships only after an out-of-sample test beats its no-script
  baseline.

### 7. QB projection: separate rushing and rushing TD channel

**Build**

- Project QB rush attempts, yards, and TDs from QB-specific player evidence and
  QB-rushing defense evidence.
- Include goal-line QB usage through observed QB rushing/TD evidence; do not borrow RB
  goal-line rates.
- Ensure player-level and defense-level tables cannot accidentally combine QB rushing
  with RB rushing.

**Acceptance criteria**

- A defense that is stout against RB carries can still project as vulnerable to QB
  rushing, and vice versa.
- Test fixtures prove no QB/RB rushing rows cross-contaminate.

### 8. RB projection: direct rushing and receiving matchup channels

**Build**

- Build distinct estimates for RB carry volume allowed, RB rushing efficiency allowed,
  RB target volume allowed, and RB receiving efficiency allowed.
- Adjust each channel for the average/expected quality of the RB offenses and players a
  defense faced, rather than classifying all opponents as a single "receiving back" or
  "runner" bucket.
- Project RB rushing/receiving TDs with conservative multi-year rate handling and
  role-specific opportunity inputs.

**Acceptance criteria**

- An RB's receiving projection can change while its rushing projection remains stable
  when only the receiving matchup changes.
- Defense tables disclose effective sample, schedule adjustment, current/prior blend,
  and uncertainty.

### 9. WR projection: target-earner and slot/wide matchup model

**Build**

- Model a WR's expected target volume using current role, route participation, target
  share, and continuous target-earner rank within the offense (WR1/WR2 context without a
  brittle label).
- Split matchup inputs by slot and wide/non-slot. Combine player alignment weights with
  the defense's alignment-specific allowed profile.
- Add depth (short/intermediate/deep) only when it is constructed from time-eligible
  play-by-play or other valid weekly data. It should layer onto alignment, not replace it.
- Keep receptions, yards, and TDs connected to target volume/efficiency but evaluate
  each component independently.

**Acceptance criteria**

- A 70% slot receiver reacts mainly to the opponent's slot profile, while a mixed player
  receives a weighted blend.
- The trace states the player's alignment mix and each defensive component used.

### 10. TE projection: inline versus slot matchup model

**Build**

- Treat inline/non-slot and slot TE usage as different roles. A blocking inline TE must
  not be projected as a slot-style receiving TE just because both carry the TE label.
- Use TE-specific slot/inline player weights and the opponent's TE alignment-specific
  allowed profile. Reuse WR mechanics only where the data definitions truly match.
- Keep TE-wide as an optional future channel; do not falsely call total-minus-slot
  "outside" when it includes inline/middle-field contexts.

**Acceptance criteria**

- TE explanation language precisely identifies known versus estimated alignment data.
- Low-route/blocking-heavy TEs cannot gain a high receiving projection solely from a
  favorable slot matchup.

### 11. Game environment and team-passing script experiments

**Build**

- Keep existing game-script behavior as a V1 benchmark; do not extend it by assumption.
- Run a focused historical study of team scoring expectation/margin against: QB attempts,
  QB passing yards, QB passing TDs, team pass rate, and skill-player volume.
- Prefer game-level/team-level evidence for team-level effects. Do not use a player's own
  prior margin split as the only evidence of team passing behavior.
- Add venue/weather only if a genuinely pregame historical forecast source is available.
  Postgame weather columns are prohibited from backtests.

**Acceptance criteria**

- Ship only metrics whose out-of-sample gain is material and stable by season/position.
- Document rejected findings with their test results, as V1 methodology already does.

### 12. Availability, injury, and conditional workload

**Build**

- Separate `plays_probability` (availability) from `workload_if_active` (conditional
  workload). The displayed expected projection can combine them, but the explanation
  must show both.
- Do not impose a generic questionable-player productivity haircut without historical
  evidence that the applicable designation and player type cause one.
- Use official/credible injury report timestamps when available. For historical tests,
  either use time-correct reports or disable availability effects and label that choice.

**Acceptance criteria**

- "Questionable, expected active" can retain a near-normal conditional workload.
- "Out" produces zero player availability and triggers the separately tested teammate
  redistribution workflow.

### 13. Vacancy and teammate redistribution model

**Build**

- QB out: identify the expected starter/back-up from an as-of roster/depth source,
  re-project team passing through that QB, then allocate targets through active route and
  target shares.
- RB out: allocate carries by expected RB snaps/carry share; allocate targets separately
  by receiving-back role.
- WR/TE out: allocate targets by expected route participation, existing target share,
  and alignment overlap. Do not assume a WR absence automatically benefits RBs.
- Include an explicit team-level "volume changes or disappears" term. Do not force a
  fixed percentage of all vacated volume onto healthy teammates.
- Allow a credible replacement to jump from a small historic snap share to a large role
  when depth, routes, and injury context support it. Avoid a rule that permanently caps
  replacements at prior usage.

**Acceptance criteria**

- Unit tests cover QB, RB, WR, and TE absences, split role distributions, and a promoted
  low-snap replacement.
- Every redistributed target/carry has a recipient rationale; unexplained volume remains
  explicitly unallocated rather than silently assigned.

### 14. Calibration and uncertainty discipline

**Build**

- Keep the raw projection visible internally and in the decomposition. Evaluate raw and
  calibrated outputs separately by position and relevant startable ranges.
- Refit/reconsider V1's one-sided calibration only on a clean train/development/test
  scheme. Do not tune repeatedly on the final holdout.
- Stress-test calibration against outlier conditions (in-game injuries, severe weather,
  early ejections) so it does not simply pull down legitimate high-confidence projections.
- Prefer calibrated intervals/uncertainty display to bluntly suppressing a well-supported
  high projection.

**Acceptance criteria**

- Calibration can be toggled without changing the raw stat line.
- Reports include MAE, RMSE, bias, rank correlation, calibration by projection bucket,
  and startable-pool results.

### 15. Projection decomposition dialog in Weekly Rankings

**Build**

- Make a Weekly Rankings table row selectable using the existing Streamlit selection
  pattern; selecting a player opens a responsive `st.dialog` (or project-consistent
  equivalent).
- The dialog receives the backend explanation payload. It must not recompute projection
  math in the UI.
- Show: projected stat line; raw and calibrated fantasy points; current/prior evidence
  and weights; expected role; opponent defensive profile; matchup components;
  availability/workload; injury/vacancy impact; game-context feature status; data
  confidence/missingness; and an "experimental/disabled" label where applicable.
- Market and FantasyPros values may appear as comparisons, clearly labeled and never
  implied to be model inputs.

**Acceptance criteria**

- Opening/closing the dialog cannot alter the ranking result.
- Selection state invalidates when year/week/scoring/model configuration changes.
- A manual browser check confirms the table stays compact and the dialog is legible.

### 16. Evaluation, release gates, and documentation

**Build**

- Preserve/rework the validation harness so every historical run is as-of-week correct,
  uses a paired player pool, and identifies source availability.
- Establish locked splits before tuning. Suggested structure: earlier seasons for fitting,
  a development season for selecting component settings, and one or more untouched later
  seasons for the release decision. Report results by week, position, and startable pool.
- Compare against V1, a simple trailing-average baseline, and a feature-ablated V2.
- Add deterministic unit fixtures for blends, alignment weighting, QB/RB rushing
  separation, defensive-prior shrinkage, injuries/vacancies, and explanation payloads.
- Update `docs/weekly_projections_methodology.md` only after a component's behavior and
  measurement are final. Include what failed or was rejected; do not turn a design
  intention into a claimed improvement.

**Release gate**

A component can become default-on only when it is leakage-safe, test-covered,
explainable, and demonstrates stable value on the locked evaluation split. Otherwise it
remains feature-gated or is removed. The user reviews the branch summary before any
GitHub push or merge.

## Recommended implementation sequence

Do not attempt all sixteen components in one agent pass. The order below minimizes the
risk of building impressive-looking UI on untrustworthy math.

1. Components 1–2: V2 shell, output contract, canonical identity/cutoff audit.
2. Components 3–4: blending and early-defense prior, with a reliable evaluation harness.
3. Components 5 and 9–10: receiver/TE alignment data contract and matchup mechanics.
4. Components 6–8: QB and RB direct channels, including QB/RB rushing separation.
5. Components 11–13: test game-script evidence, then availability and vacancy handling.
6. Component 14: recalibration/uncertainty only after the raw V2 model is stable.
7. Component 15: decomposition dialog built against the final explanation payload.
8. Component 16: release report, methodology update, user review, then decide whether to
   push or merge.

## Per-phase agent prompt template

Use this compact brief with Codex 5.3 for each phase:

> Read `docs/weekly_projection_model_v2_build_handoff.md` and the relevant V1
> methodology/code. Implement only components [N] on a new feature branch. Preserve V1
> behind feature switches, enforce historical `as_of_week` correctness, and do not use
> post-week or full-season-only data in historical tests. Before editing, state the data
> contract and tests. After editing, run the relevant unit and backtest checks, summarize
> changed files/results/known limitations, and stop before any push or merge.

## Definition of done for the full V2 effort

The model has an auditable, as-of-week-safe data foundation; position-specific role and
defensive mechanics; independently validated feature gates; raw versus calibrated values;
role-aware injury redistribution; and a usable player decomposition dialog. Every claim
of improvement is supported by locked, reproducible evaluation—not by a single promising
week or a post-hoc fit.
