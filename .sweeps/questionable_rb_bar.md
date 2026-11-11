# `v2_questionable_rb` - bar set 2026-11-11, before the run

User: "Yes, build and run" (A1 follow-up). Study: `scripts/study_questionable_wr_rb.py` -> `.sweeps/study_questionable_wr_rb.txt`.
A Questionable RB who plays gets fewer carries (0.91, 5 of 5 seasons), not worse ones (yards per carry 0.98, receiving about 1).
The flag cuts his rushing line (carries 0.927, rushing yards 0.892, rushing TDs 0.894; least squares through the origin on the 182
played rows 2022-2025, relative to untagged backs) and scores it; weeks 3-17; availability unchanged. Held out by season offline:
points MSE on those rows -1.2 (4 of 5), carries 5 of 5, rushing yards 4 of 5.

Caveat: the factors were fit on the same 2022-2025 seasons the harness scores, so the harness read is in-sample; the held-out
evidence is the offline screen above (factors fit without each season: carries 0.91-0.95, yards 0.86-0.92).

Scale: Questionable backs are about 5% of START-RB rows, so the board-wide move is small by construction (expected START-RB about
-0.004, START-ALL about -0.001). The A1 bar in docs/audit_leads_2026-11-10.md (START-ALL <= -0.003) is out of reach for a rule this
narrow; the bar below replaces it, set before the run.

**Bar** (harness v2, 2022-2025 weeks 3-17, injury + reserve replay, arm `qrb = +v2_questionable_rb` vs DEFAULT_FEATURES, 85 flags):
- the rows the arm moved (`scripts/eval_changed_rows.py`, played): RMSE delta < 0, and the carries bias moves toward 0;
- START-RB delta <= 0 and EVERY-START-RB delta <= 0;
- START-ALL delta <= +0.000 and EVERY-ALL delta <= +0.001; no other START position moves (only RB rows change).

Pass -> a ship decision for the user (DEFAULT_FEATURES 85 -> 86). Fail -> record and retire per docs/retired_flags.md.

## Result 2026-11-11 (run 12:54-13:56): PASS on every line
- Rows moved (284; played 203): played RMSE 6.214 -> 6.149 (-0.065, CI -0.19..+0.06); carries bias +0.87 -> +0.15. Played rows by
  season, points RMSE: -0.057 / -0.052 / -0.087 / -0.035 (4 of 4); carries 3 of 4, rushing yards 3 of 4 (2022 worse on both).
- START-RB 7.410 -> 7.405 (-0.005, CI -0.014..+0.004); EVERY-START-RB -0.027 (CI -0.039..-0.014), weeks 37-20.
- START-ALL -0.001 (CI -0.004..+0.002); EVERY-ALL -0.004 (CI -0.005..-0.002); EVERY-RB -0.015; QB / WR / TE unchanged (0.000).
Ship decision is the user's (DEFAULT_FEATURES 85 -> 86).
