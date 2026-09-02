# Coaching-aware defense prior — RETIRED, no signal found

**Status: CLOSED 2026-09-02.** Do not rebuild without reading "What would have to
change" at the bottom. Six independent tests over four days, three of them with
proper out-of-sample validation. The idea is dead in this form.

---

## What it was for

The weekly model blends a defense's CURRENT-season allowed profile with its
PRIOR-season one (`blend_defense_prior`, `alpha = n / (n + prior_games)`, shipped
`DEFENSE_PRIOR_GAMES = 12`). That blend treats every defense identically.

The hypothesis: **a defense that changed its coaching staff should not be
described by last season's profile as strongly as one that kept it.** A new
defensive coordinator brings a new scheme; last year's "this defense allows a lot
to slot receivers" is then partly a statement about a system that no longer
exists. So a team with a staff reset should get a SHORTER prior leash (smaller
`prior_games`, current-season data taking over faster), and a team with continuity
arguably a LONGER one.

This is intuitive, and the user's prior was that it should work. It is also
narrower than it sounds: the mechanism does not add information, it only changes
how fast one existing input decays.

Infrastructure built for it (all retained, see below):
`data/coaching_changes.py`, `data/coaching_history_wikipedia.py`, the DC-history
CSVs in `external_data/`, and the cohort classifier
(`none` / `dc_only` / `hc_only` / `both` / `dc_to_hc` / `unknown`).

---

## What we tested, and what each said

| # | Test | Question | Result |
|---|---|---|---|
| 1 | `v2_coaching_aware_defense_prior` v1 flag | Does the shipped-shape flag help? | `MAE base 4.595 vs variant 4.595` — identical to three decimals |
| 2 | Y-o-Y persistence, 2016–25 | Does a staff change actually reduce how much last year predicts this year? | **Yes, but tiny.** HC unchanged corr +0.23, HC changed +0.14. `\|d-rating\|` MAE 0.142 vs 0.160. DC-change went the *wrong* way (+0.16 → +0.23). |
| 3 | Optimal `prior_games` per cohort, OOS within-season | What blend weight does each cohort actually want? | **Incoherent.** `dc_only` wants pg24–40 (trust *longer*), `hc_only` pg2–6, `both` pg8–10. Per-season, `both`'s best flips pg20 (’23) → pg8 (’24) → pg4 (’25). |
| 4 | Position × cohort table, model A/B, wk2–9 | The one that looked promising | START-QB −0.016\*, START-TE −0.039\* (CI excl 0) |
| 5 | Confirm of #4 on wk10–17 and wk4–13 | Does it replicate? | **No.** wk10–17 START-WR **+0.020\*** (actively worse); wk4–13 flat, every CI spans 0 |
| 6 | Per-stat design sweep, 10 seasons, fit-odd/score-even | Does a cohort-specific decay shape generalise? | In-sample yes, **out-of-sample no**: every cohort's best config reverted to "no decay", improvement +0.0000 / +0.0001 / +0.0019 / +0.0000 |

\* = 95% CI excludes 0.

---

## The final verdict run (2026-09-02)

`scripts/coaching_final_verdict.py`, 2016–2025, weeks 2–10, **startable subsets
only** (the objective that actually matters for start/sit and props — whole-pool
MAE was the wrong target in tests 1–5). Four legs, built to produce a BOUND
rather than a sixth null result.

**Leg A — channel budget.** Ablate `v2_defense_prior` entirely: what is the WHOLE
prior-season defense channel worth? The coaching idea only re-weights that one
channel, so this is a hard ceiling on any refinement of it.

| pos | dMAE | 95% CI |
|---|---|---|
| QB | −0.009 | [−0.046, +0.028] |
| RB | +0.007 | [−0.017, +0.029] |
| WR | +0.011 | [−0.007, +0.028] |
| TE | **−0.058** | **[−0.101, −0.015]** |

The whole channel is worth ~nothing on startables — and for TE, removing it
*significantly helps*. **The coaching adjustment was competing for a slice of
approximately zero.** This closes the question by arithmetic, not by p-value.

**Leg B — mechanism oracle (deliberate overfitting).** Best cohort table per
position, chosen with hindsight on the very data being scored. No honest
procedure can beat this; it is the ceiling of the mechanism.

QB −0.006, RB −0.003, WR −0.001, TE −0.008 — **every CI spans zero even when
overfit.**

**Leg C — honest holdout.** Table picked on 2016–21, frozen, scored on 2022–25:

| pos | FIT dMAE | TEST dMAE | flip-acc |
|---|---|---|---|
| QB | −0.003 | **+0.003** | 0.542 |
| RB | −0.008 | **+0.004** | 0.500 |
| WR | −0.000 | −0.001 | 0.500 |
| TE | −0.013 | −0.002 | **0.444** |

The entire in-sample effect is selection. Flip-accuracy — of the player pairs the
change REORDERS, how often is the new order right — sits at a coin flip, and for
TE below one. A change that reorders players no better than chance cannot improve
start/sit whatever it does to MAE.

**Leg D — power.** Minimum detectable effect at 80% power: **QB 0.014 / RB 0.010 /
WR 0.004 / TE 0.013** points. Every one is LARGER than the Leg B oracle ceiling.

> (The run itself printed MDE = 0.0000 because it measured spread on the control
> config, which is a no-op with zero variance by construction. Fixed in `d47844c`;
> the values above are derived from the same run's bootstrap CIs.)

---

## What this told us

1. **A real descriptive effect exists.** A head-coach change genuinely cuts
   year-over-year defensive persistence from ~+0.23 to ~+0.14 correlation. That
   finding stands and is not in dispute.
2. **It does not convert to projection accuracy.** Every exploitation attempt sits
   within ±0.01 of baseline on startables, with CIs spanning zero.
3. **Every in-sample hit failed to replicate** — the wk2–9 result sign-flipped in
   wk10–17, the optimal `prior_games` flips season to season, the per-stat decay
   preference vanishes out-of-sample.
4. **The mechanism has almost no leverage.** The channel it re-weights is worth
   ~0 on startables to begin with.
5. **Sample size makes fitting hopeless anyway.** `hc_only` is ~44 team-seasons
   across a decade, `both` ~188. Split by position and season for stability and
   every cell is n < 50, against 16 parameters.

**The honest statement is a bound, not a zero:** no effect larger than ~0.01
startable points exists, and the ceiling of the mechanism sits below what this
design can detect. A real effect smaller than that may exist; it would not be
worth shipping if it did.

---

## What is retained, and why

Nothing was deleted. The descriptive finding is real and the infrastructure is
cheap to keep:

- `data/coaching_changes.py` — cohort classifier, HC history (nflverse 1999–2026),
  DC history merged from Ourlads + Wikipedia + a hand-checked manual CSV.
- `external_data/coaching_*.csv` — the DC/HC history sources.
- `v2_coaching_aware_defense_prior` — still in `MODEL_FEATURES`, **off**, so the
  wiring can be exercised without rebuilding it.
- `_POS_COHORT_DEFAULTS` — **reverted to all-`None`** (a no-op). A fitted table
  used to live there; it is disproven and the comment in that file says so. Do not
  restore it from git history without re-testing.
- `scripts/coaching_final_verdict.py`, `scripts/analyze_coaching_defense_prior*.py`,
  `scripts/sweep_pos_cohort_prior.py`, `scripts/sweep_defense_blend_design.py`.
- Raw output: `.sweeps/coaching_final_verdict.txt`,
  `.sweeps/coaching_defense_prior*.txt`, `.sweeps/sweep_pos_cohort_prior.txt`,
  `.sweeps/defense_blend_design.txt`.

---

## What would have to change to revisit

Do not re-run this with a different cohort table — that question is answered. Only
these would make it a new question:

1. **Defensive PERSONNEL continuity instead of coach names.** Snap-weighted
   returning defensive starters is the plausible *actual* mechanism — a scheme
   change matters because the people executing it changed. It is a roster signal,
   the sample is per-team-season rather than per-cohort, and we now have Ourlads
   depth charts back to 2022 to build it from. **This is the one worth doing.**
2. **A coach-quality dataset.** Everything above tests "did the staff change",
   never "is the new staff better". We have no such data.
3. **A much larger prior-season defense channel.** If `v2_defense_prior` is ever
   rebuilt into something that actually carries weight on startables, the ceiling
   in Leg A moves and the arithmetic changes. As of 2026-09-02 it does not.

---

## One live thread this produced

Leg A, plus two unrelated constant sweeps the same night, all point the same way:
**the prior-season defense channel appears to be mishandling tight ends.**

- Leg A: removing the channel entirely → TE **−0.058\*** [−0.101, −0.015]
- `PRIOR_SEASON_DEFENSE_RECENCY_FLOOR` at 1.00 → START-TE **+0.073\***
- `RECENCY_DECAY` at 0.86 → START-TE **−0.030\***

Three independent runs, same direction, on a **shipped** flag. This is NOT a
coaching finding and is not retired with the rest of this document — it is
tracked as an open item in `weekly_rankings_backlog.md`. Caveat before acting:
all three are weeks 2–10 and one scope out of four, so it needs its own confirm.
