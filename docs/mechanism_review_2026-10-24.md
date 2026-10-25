# Mechanism review: the defense extras and the strength-audit subtractions (2026-10-24)

User: "keep the PFF alignment ... The strength audit and defense extras maybe ... look into these deeper to give me a truly
conclusive explanation of what each does, its effect, and your recommendation/advice on whether it holds its own."

**How each was measured.** One full harness run (2022-2025 weeks 3-17, 60 weeks, 65-flag default with the historical injury
replay), one variant per mechanism with only that mechanism switched off, every other setting unchanged. Three readings per
mechanism:

1. **Fantasy accuracy**: change in RMSE of fantasy points when it is switched off (positive = the model is worse without it), on the
   start-worthy pool (START-ALL, ~8,100 player-games) and on every game the model expected a player to play (EVERY-ALL, 26,070), with
   a 95% bootstrap CI over weeks. The rule was written before the run (methodology 2026-10-24).
2. **Footprint**: how many players it moves and by how much, from every variant's saved board.
3. **Stat accuracy**: RMSE of each stat against what happened, on the played games it moves - where its effect lands, which the
   points scopes can hide (a mechanism can fix QB rushing yards and leave QB points unchanged).

Scale: a START-ALL RMSE is about 7.6 points, so 0.001 is about 0.01%. The harness resolves about +/-0.003 on START-ALL.

## Summary

| mechanism | what it does | players moved (avg move) | START-ALL without it | verdict | recommendation |
|---|---|---|---|---|---|
| v2_channel_matchups | nothing - never built | 0% | 0.000 exactly | dead flag | remove the name (cleanup, zero effect) |
| v2_continuous_roles | role-based defense matchup | 30% (0.12 pts) | -0.001 [-0.004,+0.002] | negligible | your call; neutral either way |
| v2_scheme_matchup | man/zone matchup for TEs | 13% (0.17; TEs 0.26) | +0.000; START-TE +0.007 | small, helps TEs | keep |
| v2_pff_defense_prior_blend | last season's PFF defense profile early on | 4% (0.10) | +0.001; START-TE +0.005 [+0.001,+0.010] | small, consistently helps | keep |
| v2_defense_blowout_discount | discounts blowout games in defense ratings | 13% (0.09) | +0.001 [-0.001,+0.002] | negligible | keep (sound, leans helpful) |
| TD matchup multiplier | defense matchup on TD rates | 40% (0.18; QBs 0.32) | +0.001; EVERY-ALL +0.002 | mixed: TD stats better without, RB points worse | keep for now; maybe position-specific |
| QB rushing game script | QB runs more when projected to trail | 4% (QBs 44%, 0.16) | +0.000 | hurts QB rushing stats | **turn off** |
| pace | opponent's plays-faced pace | 24% (0.10) | +0.002; START-RB +0.004 [+0.001,+0.007], Holm p 0.027 | **holds its own** | keep |

## Each mechanism

### v2_channel_matchups - dead flag
**What it does:** nothing. It is on in DEFAULT_FEATURES, but no line of the model checks it (MODEL_FEATURES' own comment lists it
among names "never built"). **Effect:** zero on every player, every stat, every scope. **Recommendation:** remove the name from
DEFAULT_FEATURES and MODEL_FEATURES - pure cleanup; it can only mislead someone reading the default set.

### v2_continuous_roles - the role-based defense matchup
**What it does:** rates a defense against players of the same ROLE (e.g. a team's lead receiver vs its depth receivers, a workhorse
vs a change-of-pace back), with soft role weights instead of three hard tiers, in place of the plain team-defense matchup. Switching
it off falls back to the plain matchup. **Footprint:** 30% of players, average 0.12 points, largest 1.64; stats about 1% (TDs ~3%).
**Effect:** START-ALL -0.001 [-0.004, +0.002] (removing it is neutral to a hair better); EVERY-ALL -0.000; no position outside
+/-0.003. By stat it helps TEs and RB rushing yards slightly and hurts QBs and WRs slightly, each under 0.1%. **Verdict:** negligible.
It changes many matchup reads a little and the changes cancel. **Recommendation:** your call. It adds the role-aware reading of a
matchup (like PFF alignment, context rather than accuracy); removing it simplifies the model at no cost.

### v2_scheme_matchup - man/zone matchup for tight ends
**What it does:** for TE targets, receptions and yards, replaces the broad matchup with the defense's man/zone coverage mix against
the TE's own man/zone production (computed for WRs too but scores TEs only - WR showed no start-worthy gain when it shipped).
Through pass-capacity conservation it also shifts the TE's teammates a little. **Footprint:** 27% of TEs (0.26 points), 15% of WRs
and 5% of RBs (about 0.1). **Effect:** START-ALL +0.000; START-TE +0.007 [-0.014, +0.028]. On the TE rows it moves, every stat is
worse without it: targets +0.47%, receptions +0.41%, yards +0.17%, points +0.16%. **Verdict:** small, consistently right at the
one position it scores; not significant on points. **Recommendation:** keep.

### v2_pff_defense_prior_blend - last season's PFF defense profile early in the season
**What it does:** blends a defense's completed prior-season PFF alignment and man/zone profile into its thin current-season one,
the weight fading as games accumulate (it feeds PFF alignment and the scheme matchup). **Footprint:** small - 4% of players (WR/TE),
0.10 points. **Effect:** every scope is worse without it: START-ALL +0.001 [-0.000, +0.002], EVERY-ALL +0.000 [+0.000, +0.001],
START-TE +0.005 [+0.001, +0.010] (Holm-adjusted p 0.24); on the rows it moves, TE receptions +0.59%, TE points +0.39%, WR yards
+0.09% worse without it. **Verdict:** small and consistently helpful - the best of the defense extras, just short of significance.
It also improves the input of the PFF alignment you are keeping. **Recommendation:** keep.

### v2_defense_blowout_discount - blowouts count less in a defense's rating
**What it does:** a defense's games decided by 28+ points (either way) count less when rating that defense against QBs, RBs and TEs
(garbage-time production says little about the defense). **Footprint:** 13% of players (35% of QBs), 0.09 points. **Effect:**
START-ALL +0.001 [-0.001, +0.002]; START-TE +0.004 [-0.000, +0.008]; by stat mixed and under 0.2% (QB rushing yards a hair better
without, TE points a hair worse). **Verdict:** negligible, leaning helpful at TE (it shipped on a significant START-RB win in 2024-2025
that has not repeated on four seasons). **Recommendation:** keep - principled and cheap; removing it gains nothing.

### TD matchup multiplier (tested with v2_no_td_matchup)
**What it does:** applies the defense matchup to touchdown rates (passing, rushing, receiving TDs), as it does to yards and volume.
**Footprint:** the largest of the eight - 40% of players, 0.18 points (QBs 0.32), TD stats move 6-11%. **Effect:** split. Every TD
stat is MORE accurate without it (QB passing TDs -0.70%, TE receiving TDs -0.25%, WR -0.17%, RB receiving TDs -0.13%) - the offline
strength audit's finding holds - but RB fantasy points get worse without it (+0.12%; START-RB +0.009 [-0.002, +0.019]) and the
pooled scopes lean worse (EVERY-ALL +0.002 [-0.001, +0.005], EVERY-START-ALL +0.003). **Verdict:** net negligible for fantasy points;
for TD lines it is mildly harmful except at RB. **Recommendation:** keep it for now. If you bet TD props, a position-specific version
(off for passing and receiving TDs, kept for RBs) is worth one test; the expected gain is small (0.2-0.7% on those TD stats).

### QB rushing game script (tested with v2_no_qb_rush_script)
**What it does:** scales a QB's rushing attempts and yards by the projected game script - more scrambles and designed runs when his
team is projected to trail, fewer when it leads - from a curve fit on past games. **Footprint:** 44% of QBs, 0.16 points; switching
it off lowers QB rushing a little on average. **Effect:** it makes the QB rushing line worse: without it QB rushing attempts are
2.41% more accurate and QB rushing yards 1.01% more accurate (both four of five seasons in the offline audit too), while QB fantasy
points do not move (START-QB +0.002 [-0.010, +0.013]). **Verdict:** a mechanism that hurts the stats it touches and does nothing for
points. **Recommendation:** turn it off (or delete the code path). It improves QB rushing lines - what a rushing prop and the median
view read - at no fantasy cost.

### Pace (tested with v2_no_pace)
**What it does:** scales every stat by how many plays the opponent's defense faces per game against league average, shrunk toward
neutral by games played and capped at 15% weight (v2_pace_alpha_cap). **Footprint:** 24% of players, 0.10 points, every stat about
0.7%. **Effect:** removing it hurts: START-RB +0.004 [+0.001, +0.007] with a Holm-adjusted p of 0.027 - the only significant
position result of the eight - START-ALL +0.002 [-0.000, +0.004], QB points +0.12% and QB passing yards +0.13% worse without it.
**Verdict:** holds its own (the offline audit called it inert; the full build says otherwise - another case of an offline screen
missing what a full build shows). **Recommendation:** keep.

## What this means for "leaner"

The second leanness run's `lean_defense` (all six defense extras out plus the three subtractions) came out at -0.001 because its
parts cancel: pace's cost (+0.002) and the prior blend's (+0.001) against PFF alignment's gain (-0.003) and continuous roles' (-0.001).
Taken one at a time, the model has one dead flag, one mechanism that hurts its own stats (QB rushing script), two neutral ones, and
four small helpers. Two changes are clear wins for clarity or stats at no fantasy cost: remove `v2_channel_matchups` and switch off
the QB rushing game script. Neither is shipped without the user's sign-off.

## Follow-up (2026-10-24): the TD matchup, four strengths

User: "look into td matchup ... adjust it or make a more detailed decision to get rid of it." Why it hurts TD lines: the TD stats
already carry the market's implied team total, which prices the opposing defense in, so a defense TD multiplier on top counts the
defense twice. One run on the new default, four strengths (negative = better than today):

| variant | START-ALL | every game | QB starters | RB starters | TD lines |
|---|---|---|---|---|---|
| all off | +0.0002 | +0.0015 | -0.0034 | +0.0077 | passing TDs -0.70%, receiving -0.13 to -0.25% |
| passing/receiving off, rushing half | -0.0006 | +0.0008 | -0.0043 | +0.0059 | same, rushing -0.15% |
| passing/receiving off, rushing as is | -0.0008 | +0.0004 | -0.0028 | +0.0033 | same |
| **half strength on every TD stat** | **-0.0016** (33-27 weeks) | **-0.0002** | **-0.0060** (36-24) | +0.0033 | passing -0.54%, receiving -0.18 to -0.22%, rushing -0.15% |

All four meet the bar; half strength wins it (best START-ALL, the only one better on every pooled scope, smallest moves). Turning
it off entirely makes passing-TD lines a little sharper (-0.70% vs -0.54%) but costs running backs more (+0.008 vs +0.003): the
defense TD matchup is not worthless, it is applied about twice too hard. **Recommendation: half strength (`v2_td_matchup_half`).**
