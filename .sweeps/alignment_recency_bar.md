# `v2_alignment_recency` / `_fast` - bar set 2026-11-16, before the run

User: a recency weight / detection for receiver role change, snaps and above all alignment (slot / wide / inline). Offline study
(`scripts/study_role_alignment_drift.py`, 2024-2026, WR/TE who played):
- snap / route participation: a lag exists on targets (miss +0.3-0.6 targets bottom-to-top quintile of the recent move, every
  season), but no form tried turns it into a steady points gain: an always-on ratio -0.2% (WR) / -0.5% (TE) points MSE, 2 of 3
  seasons, 2026 worse every time; detection of sustained moves (last 1-3 games, thresholds 0.10-0.25) 2 of 3 at best and the WR
  drop rule worse in absolute terms; a route-based target estimate blended in helps the whole pool (TE -0.6 to -1.0%, 3 of 3)
  but is worse on the START pool in nearly every fold. Snap-share recency already failed three times (docs/retired_flags.md).
- alignment moves: no consistent miss after a sustained slot / wide / inline move in any season.

The flags: the alignment mix the alignment matchup reads weights each game by 0.85 ** games_ago (`v2_alignment_recency`, the
model's own per-game decay) or 0.70 (`_fast`) instead of season-to-date snaps. Smoke (2025 week 12, 0.70): 20 of 497 rows move,
mean 0.12 pts, max 0.4.

**Bar** (harness v2, 2024-2025 weeks 3-17 - the seasons with weekly PFF; injury + reserve replay; arms `ar85`, `ar70` vs
DEFAULT_FEATURES, 86 flags):
- ship candidate only if START-WR <= -0.002 and START-TE <= 0, EVERY-WR and EVERY-TE <= 0, START-ALL <= 0 (the better arm);
- otherwise record as inert / failed in docs/retired_flags.md and remove.
Expected: inside +/-0.002 everywhere.

## Result 2026-11-16 (run 08:47-09:53): FAIL (inert), as expected - removed from the code
- ar85: START-WR -0.000 (CI -0.002..+0.001), START-TE +0.001, START-ALL +0.000, EVERY-WR / EVERY-TE / EVERY-ALL 0.000.
- ar70: START-WR -0.001 (CI -0.003..+0.001), START-TE +0.000, START-ALL -0.000, EVERY-WR / EVERY-TE +0.000.
- Rows moved (`scripts/eval_changed_rows.py`, played): ar85 246 rows, WR +0.009 (CI -0.011..+0.025), TE -0.004; ar70 462 rows,
  WR +0.006, TE -0.003. Every moved row was lowered by about 0.1 point.
Recorded in docs/retired_flags.md ("After wave 2"); code in archive/retired_flags/patches/18-uncommitted-alignment-recency.patch.
