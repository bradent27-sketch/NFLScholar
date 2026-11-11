# Prop tiers v0 - definitions and bar (written 2026-11-11, before any tier was scored)

Audit lead B2 (docs/audit_leads_2026-11-10.md). Tiers label a priced prop; they never move a projection. Changing any
definition below makes a v1 whose record starts from zero.

## Inputs, per priced line
The sharpest book's priced line (data.odds_projections.LINE_BOOKS order, lines carrying a de-vigged p_over), at the time the
board was built: stat, model P(over) `p` (data/prop_probability.py, raw), the book's no-vig P(over) `q`, both prices.

- Model side: over when p > q, else under. Gap = |p - q|.
- Price lean: over when q > 0.5, under when q < 0.5, none at exactly 0.5. "With the lean" = model side equals the lean.

## Tiers (first match wins)
1. **No bet:** stat is passing yards, passing attempts, passing completions, passing TDs, interceptions or carries
   (rushing_attempts) - their B8 switch is off until B8 says otherwise; or gap > 0.25. (B8 row flags: none exist in v0.)
2. **A:** stat is receiving yards or receptions, or a rushing-yards UNDER; with the lean; 0.03 <= gap <= 0.25.
3. **B:** the same stats and sides as A, against the lean or with no lean; 0.03 <= gap <= 0.25.
4. **Track:** everything else (gap under 0.03, rushing-yards overs, targets, any other stat). Shown, never recommended.

## Shrunk probability and EV (reported, does not decide a tier)
p' = 0.5 + s_stat (p - 0.5). s_stat = data.prop_record.best_shrink on the 'at close' sharpest-book rows of weeks before w for
that stat when it has 200+ rows, else on all stats' rows when 300+, else 0.34 (DEFAULT_SHRINK). EV of the side taken =
p'_side x decimal price - 1.

## Record per tier (forward-chained: week w's shrink uses weeks before w; the tier rules are fixed)
bets; hit rate with a 95% Wilson interval; the hit rate shown shrunk toward 50% as (hits + 100) / (bets + 200); ROI at the
posted price of the side taken, with a 90% lower bound (mean - 1.645 sd / sqrt n); closing-line value: share of bets the close
moved toward (an over: the closing line higher, or the same line with a higher no-vig P(over); an under the mirror), over the
bets that had a close from the same book.

## Bar
A tier reads **bettable** only with 300+ forward bets AND a 90% lower bound on ROI at the price above 0. Otherwise it reads
**tracking**. Weeks scored: every week with a pre-kickoff board and archived priced lines (2026 weeks 3-9 today).

## Display
One TIER column on the game view's priced-props table (A / B / Track / No bet; blank for a line with no priced book), tiered
rows sorted first (A, B, Track, No bet, then the existing largest-gap order). The tier record goes on the Track Record page.
No other board, column or game-view change.

## Amendment (2026-11-11, same session, before any week-10 result exists; stricter, not looser)
The v0 rules were chosen from what weeks 3-8 showed (B3's read), so weeks 3-9 cannot test them. The status (bettable /
tracking) counts only boards from 2026 week 10 on (`data.prop_tiers.FORWARD_FROM`); weeks 3-9 are shown as in-sample
background. In-sample, weeks 3-9: A 738 bets 57.6% hit, ROI at price +1.3% (90% low -4.0%); B 1,174 bets 51.8%, +3.1%
(-1.7%); Track 662, 45.8%, -13.0%; No bet 1,243, 47.7%, -9.2%.
