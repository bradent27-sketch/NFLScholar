# `v2_venue_mult` — RETIRED, no measurable effect

**Status: CLOSED 2026-09-02.** Built, gated, unit-tested, and — once finally
backtested — measurably not helping. Code retained, flag stays off.

---

## What it was

An **indoor/outdoor venue multiplier**: scale a player's projected stats by
whether the game is played in a dome, a retractable roof, or outdoors.

It came from unbundling `game_env`, a bundle rejected on 2026-08-27 for
costing +0.012 MAE. That bundle contained two independent ideas welded
together, so it was split into its two real parts:

| flag | idea | outcome |
|---|---|---|
| `v2_game_total_elasticity` | scale volume by the market's implied game total | **SHIPPED 2026-08-31** |
| `v2_venue_mult` | scale by indoor/outdoor venue | **retired here** |

The unbundling was the right call — one half was genuinely good. This is the
other half.

---

## Why it sat untested so long

It was the last flag in `MODEL_FEATURES` with **no standalone backtest at
all**. It had passing unit tests and live gating code, which is exactly the
state that reads as "nearly ready" and isn't: tests prove the code does what
it says, not that what it says is worth doing. It sat in the backlog as the
"#1 hanging item" through several sessions on that basis alone.

---

## The result

`scripts/backtest_component.py --add v2_venue_mult`, 2021–2025 weeks 3–17,
n=22,924 player-weeks over 75 scored weeks. Positive dMAE = adding the flag
makes projections WORSE.

| scope | n | dMAE | 95% CI | weeks won-lost |
|---|---|---|---|---|
| ALL | 22,924 | **+0.003** | [−0.001, +0.007] | 33–42 |
| QB | 2,378 | +0.010 | [−0.007, +0.028] | 36–39 |
| RB | 6,018 | +0.001 | [−0.003, +0.005] | **28–47 (p=0.04)** |
| WR | 9,699 | +0.002 | [−0.003, +0.006] | 38–37 |
| TE | 4,829 | +0.004 | [−0.002, +0.011] | 31–44 |
| START-QB | 1,657 | +0.028 | [−0.011, +0.064] | 34–41 |
| START-RB | 2,570 | −0.004 | [−0.027, +0.020] | 41–34 |
| START-WR | 3,513 | +0.012 | [−0.016, +0.036] | **27–48 (p=0.02)** |
| START-TE | 1,288 | +0.019 | [−0.029, +0.065] | 36–39 |

**Eight of nine scopes are positive (worse).** No CI excludes zero, so no
single scope is individually significant — but two sign tests are: RB lost 47
of 75 weeks (p=0.04) and START-WR lost 48 of 75 (p=0.02). A feature that is
directionally worse everywhere and significantly loses more weeks than it
wins in two scopes is not a marginal call.

---

## Why it doesn't work, most likely

**The information is already in the model.** `v2_game_total_elasticity`
shipped the day before this was tested, and it scales volume by the market's
implied game total. A dome's scoring environment is *already priced into that
total* — Vegas knows the game is indoors. So the venue multiplier is a second,
cruder helping of a signal the model now takes from a sharper source, and
double-counting it costs accuracy.

That also explains why the original `game_env` bundle measured worse than
either half suggested: the elasticity half was carrying the bundle while the
venue half dragged on it.

---

## What is retained

- `v2_venue_mult` stays in `MODEL_FEATURES`, **off**, with its gating code
  (2 call sites) intact, so it can be re-tested without rebuilding.
- Raw output: `.sweeps/venue_mult_standalone.txt`.

## What would have to change to revisit

Only one thing would make this a new question: **dropping or materially
weakening `v2_game_total_elasticity`**. If the model ever stops taking the
scoring environment from the implied total, venue becomes an independent
signal again rather than a duplicate, and the arithmetic changes. As long as
that flag ships, this one is answered.

Do not re-test this by tuning the multiplier's strength. The problem is not
its depth, it is that the channel is already occupied.
