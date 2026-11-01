# Deep audit (2026-10-28)

User: "build a memory intensive test to work through the next 4-5 hours ... whatever kind of testing you can come up with that can
help identify flaws, strengths, new factors we can't currently see etc. This should have the capability of building prop specific
accuracy for prop betting going forward."

`scripts/deep_audit.py` rebuilt every board of today's model (the 76 DEFAULT_FEATURES plus the harness injury/reserve replay)
for 2021-2025 weeks 1-18 and 2026 weeks 1-7: 97 boards and 48,609 player-weeks. Each row is joined to:
- the actual box score;
- the player's history before that week;
- the opponent's allowed rates;
- team volume;
- the game's spread, total, weather, rest and kick time;
- the teammates who were out.

It ran in two passes:

- Pass 1 (69 minutes): baselines, segments, a boosted residual search and four prop distributions.
- Pass 2 (21 minutes): it followed pass 1's leads:
  - week windows;
  - how much of the model's departure from a player's own level comes true;
  - prop engines on recalibrated means.

The PC was free longer than this needed; see "Next" for what is worth queuing.

Graded rows are played games (a box score or a silent game), weeks 3-17 unless stated. QBs are the projected starter. Tables are
in `.sweeps/deep_audit/REPORT.md` and `REPORT_PASS2.md`.

## 1. Strengths

- **The model beats every simple baseline on every stat.** Against the player's own season to date it removes 7-20% of the
  squared error on most stats, against last 3 games 13-27%, and against last game 33-49%.
- **Fantasy points are clean.**
  - A boosted model of the points residual, given ~245 features, finds nothing held out: gains of -0.5% to +0.4% against a
    shuffled-residual null.
  - The points layer (lines, anchor, luck) already absorbs what the stats get wrong.
  - Stacking the player's own history on the points adds at most 0.8% (TE).
- **Game context is handled.** Weather (wind, temperature), dome, prime time, rest, divisional games, surface and weekday show no
  material bias once false discoveries are controlled. Of 8,256 bins, the context ones each remove under 0.1% of the error.
- **QB passing departures are right.** When the model moves a QB's passing volume away from his own average, about 100% of the
  move comes true (k = 0.99 attempts, 1.00 yards).
- **Touchdown projections are calibrated.** For anytime TD, a Poisson on the projected TDs scores as well as a fitted logistic and
  a boosted classifier: Brier 0.177-0.190, calibration error 2-4%.

## 2. Flaws, largest first

1. **The stat projections overshoot when they leave a player's own level (the main finding).**

   Here k is the share of the gap between the projection and the player's own average (season to date after 3 games, else last
   season) that actually came true. Weeks 3-17, rows projected 3+ points:

   | | k | held out (k fit on other seasons): MSE gain, 2021-2025 / 2026 |
   |---|---|---|
   | QB rushing attempts / yards | **0.43 / 0.51** | 5.4% / -0.9%; 2.5% / -0.9% |
   | TE targets / receptions / yards | **0.58 / 0.58 / 0.71** | 2.7% / 5.1%; 2.4% / 2.5%; 1.0% / -0.5% |
   | WR targets / receptions / yards | **0.65 / 0.70 / 0.77** | 2.3% / 1.6%; 1.3% / 2.0%; 0.7% / -0.2% |
   | RB carries / targets / receptions / rec yards / rush yards | 0.75 / 0.76 / 0.78 / 0.81 / 0.85 | 1.7% / 1.7%; 0.9% / 1.4%; ... |
   | QB attempts / passing yards / passing TDs | 0.99 / 1.00 / 0.97 | none |
   | Points: QB / RB / WR / TE | 0.96 / 1.03 / 0.89 / 0.83 | 0.0 / -0.1 / 0.1 / 0.4% (2026: -0.8 / 0.1 / 0.1 / -0.1%) |

   - Every season agrees.
   - It holds both ways for WR/RB (up and down departures).
   - For TEs the cut below their own level is the bigger miss: k 0.53 down vs 0.78 up. This is the known "TEs run under" (L4).
   - For QB rushing it is the raise: up 0.33, down 0.99.
   - It holds with and without a teammate out (WR 0.60 / 0.73), early and late, for favorites and underdogs, and against soft and
     hard defenses. So it is not one mechanism (vacancy, matchup, game script). The stat projections are simply too wide around
     each player's level.
   - The points layer compresses this for fantasy points. It does not touch the stat columns that props are priced on.
2. **QB rushing is barely better than the QB's own average.**
   - Skill against season to date: 0.4% (attempts), 2.7% (yards).
   - The boosted residual model finds the most here: 7.4% and 5.1% held out, every season.
   - Favorites by 7.5+ are over-projected by about 6 rushing yards.
3. **Week 18 (the untested window, weakness 5):**
   - The QB points bias is +2.1, against +0.3 to +0.6 in other weeks.
   - QB passing yards are +27 and attempts +3.2.
   - The slope of actual on projected falls to 0.75-0.84 at every position.
   - Collapses (a 10+ projection scoring under a quarter of it) are 12-13% for RB/WR, against 5-8%.
   - These are rested starters. Fixing it needs clinch/seeding data.
4. **Week 1:**
   - Receiving yards run high: WR +4.6, TE +3.0, RB +2.1. WR/TE points run +0.6.
   - For RB/WR the model is no better than last season's average (skill -0.03 / +0.005).
5. **QB passing yards in weeks 9-17 run +6.6 high** (all seasons). Late-season passing declines more than the board allows.
6. **Interceptions:** the slope of the result on the projection is 0.15. INT is mostly noise, and the spread of INT projections
   is far too wide. Across QBs, attempts (0.68) and TDs (0.71) are also too wide. A QB's departures from his own level are right;
   the gaps between QBs are what is too wide.

**The high calls (progress summary weakness 1) are not this overshoot.** On the 2026 live boards, where the model sits 2+ points
above the market (114 rows):
- the model is +0.5 above the player's own level, the market 2.5 below it;
- actual 10.6, market 10.3, model 13.3.

The market knows something history doesn't (news, role, health). Shrinking toward the player's level leaves those rows where they
were: 35% closer than the market, as before. The high-call audit should look at information (injury and role news), not the
spread.

## 3. New factors

Beyond the projection itself, the boosted search's consistent features are all small. Each removes at most 1.4% of a stat's
residual:
- the player's volatility (QB rushing sd);
- recent efficiency (rushing/receiving EPA, first downs);
- the listed injury status for WR receiving yards (6 of 6 seasons);
- the implied team total for QB points;
- the game total for QB passing yards.

Each stat residual's top feature is the stat's own projection. That is the overshoot above, seen from the other side. No large
unseen factor turned up.

## 4. Props

**Shape.**
- The Poisson / Normal the de-vig code assumes is too narrow for volume stats. Its 50% interval held:

  | stat | share of results inside |
  |---|---|
  | passing attempts | 35% |
  | carries | 36% |
  | targets | 41% |
  | receptions | 43% |
  | passing yards / receiving yards | 50-53% (right) |

- An empirical spread (the actual-to-projection ratio by projection level) and a fitted negative binomial / zero-inflated gamma
  are both calibrated: 48-53% and 78-83% coverage, PIT within 2-5%.
- Side effect: `data/market_devig.py` solves the market's implied mean for counts with a Poisson. Its 'Mkt' implied means for
  attempts, targets and receptions are therefore slightly off at prices far from even.

**Engines (forward-chained 2023-2026; CRPS change against the empirical shape on the raw projection):**

| stat | recalibrated mean + empirical (shipped) | boosted mean | quantile boosting, all features |
|---|---|---|---|
| targets | -0.9% | -1.2% | -1.6% |
| receptions | -0.8% | -0.9% | -0.4% |
| receiving yards | -0.3% | -0.7% | -0.8% |
| rushing attempts | -0.9% | -1.1% | -0.4% |
| rushing yards | -0.2% | +0.8% | +1.3% |
| passing yards | -0.5% | -1.6% | -0.8% |
| passing TDs / INT | -0.1% / +0.8% | +0.8% / +0.5% | +1.8% / +4.1% |

- "Recalibrated mean" is a ridge fit, per position, of the result on the projection, the season to date, the last 3 games and
  last season.
- The boosted versions gain a little more on some stats but need the whole feature panel at run time.
- The recalibrated mean needs only the player's box scores, so it is the engine that shipped.
- QB attempts, completions and interceptions ship without the recalibration: it lost there (CRPS +0.2 to +0.8%). The empirical
  tables already correct the level bin by bin.

**Against the books (2026 weeks 3-7, closing lines from Pinnacle, then DraftKings, de-vigged).** The engine below is fit on
2021-2025 only, applied to the live pre-kickoff boards (2,684 lines):

| | Brier model / book | best shrink toward 1/2 | 5%+ edge: bets, hit [95% CI], ROI at -110 |
|---|---|---|---|
| all | 0.255 / 0.248 | 0.34 | 1,607, 52.3% [49.9, 54.8], -0.1% |
| receiving yards | 0.253 / 0.250 | 0.36 | 467, **55.0%** [50.5, 59.5], +5.1% |
| receptions | 0.253 / 0.247 | 0.42 | 519, **54.3%** [50.0, 58.6], +3.7% |
| rushing yards | 0.253 / 0.251 | 0.40 | 221, 53.8% [47.3, 60.3], +2.8% |
| rushing attempts | 0.265 / 0.249 | 0.16 | 150, 52.0%, -0.7% |
| passing TDs / INT | 0.236 / 0.235; 0.245 / 0.243 | ~1 | 67, 49%; 41, 46% |
| **passing yards / attempts** | 0.266 / 0.251; 0.275 / 0.248 | 0 | 54, **37%**; 88, **38%** |

- **The books' price is the better probability**, and the model's P(over) carries about a third of the confidence it shows.
- But its side still wins on receiving yards and receptions:
  - Together with rushing yards, 54.5% [51.7, 57.3] on 1,207 bets at 5%+, +4.1% at -110.
  - Unders hit 55.9%, overs 52.0%.
  - Week 3 hit 48%; weeks 4-7 hit 52-61%.
  - This agrees with the earlier lean record (receiving yards, gap 10+: 62%).
  - Five weeks and slices chosen after the fact: promising, not proven.
- **Passing yards and attempts lose badly in 2026 (37%).** This is W1, the 2026 passing volume. Don't bet them until W1 clears.
- The rebuilt boards (no live injury pulls) do worse than the live boards: 50.9% at 5%+. Much of the edge is the live injury and
  role information on the boards the user actually built.

## 5. What was built (tooling; no model flag changed)

- **`data/prop_probability.py`**: P(over a line) for a board stat. It recalibrates the projection with the player's history,
  then reads the empirical spread. Engines are in `data/prop_engines.json` (fit 2021-2026). It reproduces the audit's engine
  exactly on the mean and within 0.005 on P(over). Tests: `tests/test_prop_probability.py`.
- **`scripts/prop_scorecard.py`**: the weekly prop record on the live boards against priced lines, at board time and at close:
  - Brier against the book;
  - calibration bands;
  - the best shrink;
  - hit rate and ROI by edge, per stat.

  `--sheet WEEK` writes the upcoming week's sheet (`.sweeps/prop_sheet_<year>_wkNN.csv`): the recalibrated mean, P(over), P(over)
  shrunk by the record's factor, the book's price and the gap.
- **`scripts/deep_audit.py`**: the audit itself, re-runnable by stage:
  - `--stage all` runs pass 1;
  - `--stage pass2` runs pass 2;
  - `--stage engines` refits the prop engines on all seasons and on 2021-2025.

  Refit the engines after each season (or once 2026 has 8+ more weeks):
  1. `--stage all`
  2. `--stage engines`
  3. `python scripts/prop_scorecard.py --install-engines`

Weekly use:

```
python scripts/prop_scorecard.py              # the record so far
python scripts/prop_scorecard.py --sheet 9    # next week's sheet from the latest week 9 board
```

## 6. Next (each needs the user's call)

1. **Board stat recalibration flag** (`v2_stat_recal_own_history`, candidate). Pull each stat column toward the player's own
   level by the measured k. This would make the app's stat columns, games view and leans match the prop engine. Fantasy points
   would not change: the points layer already handles it, and the shrink tested flat on points.
2. **Prop view (P1/P2).** P1's calibrated stat lines now exist as tooling. A prop page would show the sheet in the app.
3. **QB rushing.** Under half of the model's departures from a QB's own rushing level come true. Item 1 covers the stat
   columns; on points (QB k 0.96) there is nothing to do.
4. **Week 18 rule**, needs standings/clinch data. **Week 1**: receiving yards high.
5. **High-call audit, refocused on information**: injury and role news the market prices first. It is not the spread.
6. Queue for a free evening: a harness screen of item 1's points-neutral version on the stat-based scopes, and the weekly prop
   record once weeks 8-9 have results.
