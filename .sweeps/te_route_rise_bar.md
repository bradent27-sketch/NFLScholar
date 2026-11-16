# `v2_te_route_rise` / `v2_wr_route_rise` - bar set 2026-11-16, before the run

User: a recency weight / detection for receiver role change (snaps, alignment; WR and TE). The study
(`scripts/study_role_alignment_drift.py`, 2024-2026 WR/TE who played) found one form that holds up: a tight end whose route
participation (PFF weekly routes / team dropbacks) jumps in his latest game is under-projected the next week. The board reads
targets and snaps, not routes. Rule: last game's participation 0.15+ above his pooled earlier-season level (2+ earlier games for
the team) -> targets / receptions / receiving yards / receiving TDs x min((rp_last / rp_before) ** 0.35, 1.5); weeks 3-17; off when
his latest weekly export is not his team's latest game. Offline, held out by season (`.sweeps/study_role_alignment_drift.txt`,
Detection): the moved rows' points MSE -3.7% / -1.7% / -0.7% (2024 / 2025 / 2026), whole TE pool 3 of 3, START-TE -0.5% /
-1.1% / +1.3% (the 2026 rows were the live ledger, built by older models). Drops are not acted on (worse at every threshold,
WR and TE). Every alignment move failed offline.

Second arm, `v2_wr_route_rise`: the same rule for WRs at 0.20+, exponent 0.30 (held-out fits 0.20 / 0.30 / 0.35). Offline the
moved rows -2.9% / -3.3% / +2.0%, START-WR -0.2% / -0.3% / +0.0% - 2 of 3, the miss in the confounded 2026 fold, which the
2026 rebuild here decides.

Smoke (2025 week 12): TE arm 7 of 497 rows move, all TE, +0.4 to +1.6 points (Brock Wright, Bowers, Kmet, Helm, Knox, Moreau,
FitzPatrick); WR arm 12 rows, all WR, up to +2.0 (Burden, Vele, Tim Patrick, Dortch, Hopkins).

Caveat: the threshold and exponent were chosen on these seasons (the exponent fit without each season 0.25-0.45), so the
harness read is near in-sample; the 2026 fold is rebuilt with today's model, which the offline study could not do.

**Bar** (harness v2, injury + reserve replay, arms `trr = +v2_te_route_rise`, `wrr = +v2_wr_route_rise` vs DEFAULT_FEATURES,
86 flags; runs 2024-2025 weeks 3-17 and 2026 weeks 3-10). TE arm:
- the rows the arm moved, played (`scripts/eval_changed_rows.py`): RMSE delta < 0 pooled over the three seasons and in at least
  2 of 3 seasons;
- EVERY-TE delta <= 0 in both runs;
- START-TE delta <= 0 in 2024-2025, and <= +0.010 in 2026 (n about 20 a week);
- START-ALL delta <= +0.001 in both runs; only TE rows move.

WR arm, the same lines with WR for TE (START-WR <= 0 in 2024-2025 and <= +0.005 in 2026, n about 55 a week); only WR rows move.
Each arm is judged on its own; both passing -> two separate ship decisions.

Pass -> a ship decision for the user (DEFAULT_FEATURES 86 -> 87 per flag; it then joins POST_CALIBRATION_FEATURES). Fail -> record
and retire per docs/retired_flags.md.

## Result 2026-11-16 (run 10:02-11:10): FAIL, both arms, on the 2026 fold - removed from the code
TE arm `trr`:
- 2024-2025: START-TE -0.009 (CI -0.042..+0.025), EVERY-TE -0.003, START-ALL -0.002, EVERY-ALL -0.001 - passes.
- 2026 weeks 3-10: START-TE +0.025 (CI -0.032..+0.081; limit +0.010), EVERY-TE +0.012 (limit 0), START-ALL +0.003 (limit +0.001).
- Moved played rows, points RMSE: 2024 -0.142 (114), 2025 -0.099 (116), 2026 +0.020 (41; eval_changed_rows played +0.127) -
  pooled better, 2 of 3. Mean projection base / arm / actual: 2024 6.36 / 7.33 / 7.80, 2025 6.04 / 7.15 / 6.96, 2026 6.74 /
  7.84 / 7.21: the moved TEs beat the board every season, but by 2026 the board (today's model) already carries most of it
  and the full bump overshoots.
WR arm `wrr`:
- 2024-2025: START-WR -0.003 (CI -0.018..+0.012), EVERY-WR -0.005, START-ALL -0.001 - passes.
- 2026: START-WR +0.011 (limit +0.005), EVERY-WR +0.008 (limit 0), START-ALL +0.004.
- Moved played rows: 2024 -0.147 (145), 2025 -0.136 (145), 2026 +0.007 (79). Base / arm / actual 2026: 6.10 / 7.09 / 7.06.
Not re-tuned: a smaller exponent would be fit on the fold that just failed. Recorded in docs/retired_flags.md ("After wave 2");
code in archive/retired_flags/patches/19-uncommitted-te-wr-route-rise.patch (applies cleanly to 17a43cb).
Side read: the 2026 rebuild's TE base bias is high - START-TE +1.10, EVERY-TE +0.09 (2024-2025: -0.34 / -0.12).
