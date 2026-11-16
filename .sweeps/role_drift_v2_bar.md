# Role drift, second round - bars set 2026-11-16, before the run

User (after round one): find the best alignment decay and keep it even as an accuracy push; keep chasing why the route rise
worked in 2024-2025 and not in 2026, valuing 2026 slightly less (smaller sample).

## `v2_alignment_recency` (WR 0.80, TE 0.85)
The decay is chosen by how well it predicts the player's next game's alignment mix (2024-2026, decays 0.40-1.00): WR best 0.80
(slot and wide, 2024 and 2025 both 0.80, 2026 0.95), TE 0.85 (slot and in line; wide 0.90). Round one: 0.85 / 0.70 inert.
Bar (a push is acceptable, harm is not): in both runs START-WR, START-TE, START-ALL, EVERY-WR, EVERY-TE <= +0.001.

## `v2_te_route_rise` (0.15+, exponent 0.20) / `v2_wr_route_rise` (0.20+, exponent 0.15)
Why round one failed in 2026 (harness base boards, today's model, all three seasons; scratch rr_diag.py):
- TE: in 2024-2025 the board under-projected the whole TE pool (+0.56 / +0.53 a played row, weeks 3-10), 2026 does not (-0.11).
  Against that pool miss the moved TEs beat the board by the same amount every season (+0.88 / +0.62 / +0.57), less than the
  1.05 bump: the 0.35 exponent had partly learned "raise any TE".
- WR: weaker in weeks 3-10 in every season (2024 dMSE +0.97, 2025 -0.97; weeks 11-17 -5.2 / -2.0); 2026 has weeks 3-10 only.
  Pool-relative, the 2026 moved WRs beat the board by +0.28 (2024 +0.70, 2025 +1.34).
- 2026 rises fade a little faster (TE keeps a median 47% of the jump next game vs 58% / 71%); more earlier games (3-5)
  does not help.
Refit: exponent fit on the miss relative to the position pool (same season, weeks 3-10 / 11-17), held out by season:
TE 0.20 / 0.25 / 0.20 -> moved rows -3.1% / -2.7% / -1.6%, TE pool -0.75% / -0.49% / -0.13%, START-TE -0.45% / -0.76% / +0.22%;
WR 0.15 / 0.10 / 0.20 -> moved rows -2.3% / -2.0% / -0.8%, START-WR -0.15% / -0.08% / -0.06%, WR pool 3 of 3.
Caveat: the pool-relative idea came from reading the 2026 rebuild, so 2026 is not a clean fold; the 2026 exponent itself was
fit on 2024-2025 only.

Bar (harness v2, injury + reserve replay, arms vs DEFAULT_FEATURES 86; 2024-2025 weeks 3-17 and 2026 weeks 3-10), each arm alone:
- moved played rows (`scripts/eval_changed_rows.py` / per season): RMSE better pooled and in at least 2 of 3 seasons;
- EVERY-pos <= 0 in 2024-2025 and <= +0.003 in 2026;
- START-pos <= 0 in 2024-2025 and <= +0.010 (TE) / +0.005 (WR) in 2026, and the row-weighted pool of the two runs <= 0;
- START-ALL <= +0.001 in both runs; only that position's rows move.
Pass -> a ship decision for the user (one flag per change; it then joins POST_CALIBRATION_FEATURES). Fail -> retire again.

## Result 2026-11-16 (run 11:49-13:55)
`al` (WR 0.80 + TE 0.85) - FAIL narrowly, on TE:
- 2024-2025: START-WR -0.000, START-TE +0.001, START-ALL +0.000, EVERY-WR +0.000, EVERY-TE +0.000.
- 2026: START-WR -0.002 (CI -0.004..+0.000), START-TE +0.002 (CI +0.000..+0.004; limit +0.001), START-ALL -0.000, EVERY-WR -0.000,
  EVERY-TE +0.000. Moved played rows 2024 +0.029 (143), 2025 -0.014 (183), 2026 -0.012 (36), pooled +0.003: a push.
`trr` - FAIL on one line (2026 EVERY-TE):
- 2024-2025: START-TE -0.017 (CI -0.037..+0.003), EVERY-TE -0.006, START-ALL -0.003 - passes.
- 2026: START-TE +0.007 (CI -0.024..+0.037; limit +0.010) passes, EVERY-TE +0.005 (CI -0.009..+0.017; limit +0.003) FAILS,
  START-ALL +0.001 passes. Row-weighted pool START-TE -0.012, EVERY-TE -0.004.
- Moved played rows: 2024 -0.087 (120), 2025 -0.045 (127), 2026 +0.034 (46), pooled -0.054 - 2 of 3, passes.
  Base / arm / actual: 2024 6.19 / 6.73 / 7.41, 2025 5.88 / 6.49 / 6.36, 2026 6.39 / 6.98 / 6.42: in 2026 the moved TEs were
  already projected right, so any bump overshoots (round one, exponent 0.35: 2026 EVERY-TE +0.012 - the refit halved it).
`wrr` - PASS, every line:
- 2024-2025: START-WR -0.005 (CI -0.012..+0.002), EVERY-WR -0.004, START-ALL -0.002.
- 2026: START-WR +0.001 (limit +0.005), EVERY-WR +0.002 (limit +0.003), START-ALL +0.000. Pool START-WR -0.004.
- Moved played rows: 2024 -0.068 (156), 2025 -0.073 (155), 2026 -0.001 (83), pooled -0.055 - 3 of 3. Only WR rows move.

## `v2_alignment_recency` WR only (0.80) - bar set before the run
The TE decay cost START-TE +0.001 / +0.002 in both runs (not one fold), so it is dropped; TEs keep the season-to-date mix.
Arm `alw`. Bar unchanged: in both runs START-WR, START-TE, START-ALL, EVERY-WR, EVERY-TE <= +0.001.

## Result, WR only (run 13:50-14:28): FAIL by 0.001 on one line, same as `al`
- 2026: START-WR -0.002 (CI -0.004..-0.000), START-TE +0.002 (CI +0.000..+0.004), START-ALL -0.001, EVERY-WR -0.000, EVERY-TE +0.000.
- 2024-2025: START-WR -0.000, START-TE +0.001, START-ALL +0.000, EVERY-* +0.000.
Identical to WR + TE, so the TE decay moves no points. The START-TE cost is the WR mix shifting team targets: 3 TE rows in 2026
moved 0.1-0.2 (Tyler Warren week 8 +0.2 most of it). The TE decay (0.85) is restored; the flag is a push, as the user expected.
Why 2026 read worse for the route arms (base boards, played rows; moved-row miss = actual - base, weeks 3-10 / 11-17):
- WR: against the WR pool, the moved rows beat the board only in weeks 11-17 (2024 +1.67, 2025 +1.38); weeks 3-10 are a wash in
  every season (-0.31 / +0.26 / +0.02). The 2026 rebuild has weeks 3-10 only.
- TE: 2026 moved rows miss +0.03 (SE 0.66, n 46) vs 2024-2025 weeks 3-10 +0.99 (n 106): z = -1.1, within noise. Each
  season-half's signal is about 1 SE, so a single fold can read flat.

## Decision 2026-11-16: all three SHIPPED by the user (DEFAULT_FEATURES 86 -> 89)
"Alignment recency push .8 and .85. Wr route rise push 0.15. Te rout rise, push 0.2" - the TE arm over its one missed 2026 line
(the user values the smaller 2026 sample less). Live smoke 2026 week 11: 6 WR rows move (max +0.6, Tez Johnson); the route arms
wait for the week-10 PFF weekly export (data through week 9; the guard keeps them off until a team's latest game is in it).
