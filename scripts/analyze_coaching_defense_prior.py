"""Does a defensive coaching change weaken last season's defense prior?

The weekly model blends a team's PRIOR-season defense-allowed profile into its
current-season one with `blend_defense_prior`: alpha = n / (n + DEFENSE_PRIOR_
GAMES), n = current-season defensive games so far, DEFENSE_PRIOR_GAMES = 12.
That single constant assumes every defense carries its identity forward from
year to year equally well. Hypothesis (user, 2026-08-30): a team that changed
its defensive staff - especially its coordinator - should get LESS leash to
last year's profile; a team that kept its staff intact maybe deserves MORE.

This script measures it two ways, per coaching cohort
(none / dc_only / hc_only / both; see data.coaching_changes):

  1. YEAR-OVER-YEAR PERSISTENCE. For each defensive (team, season Y), a
     per-position defense-allowed rating (PPR points allowed to that position
     per game / that season's league mean for the position, so ~1.0 = average,
     <1 = stingy). Correlate rating_{Y-1} vs rating_Y across teams, split by
     cohort. Low persistence after a DC change => the prior is stale => it
     should be down-weighted.

  2. OPTIMAL PRIOR WEIGHT, within season. For a grid of prior_games values and
     each week cutoff n, blend rating_{Y-1} with the team's own weeks-1..n
     rating (weight n/(n+prior_games)) and score it against the team's
     ACTUAL weeks-(n+1)..17 rating (strictly out of sample). Report the
     prior_games that minimises blended MAE, per cohort. If the no-change
     cohort wants a bigger prior_games than the DC-change cohort, the model
     should make DEFENSE_PRIOR_GAMES coaching-aware.

HC data is nflverse (1999-2026, reliable). DC data is the Ourlads archive
(2022-2025 only) - so cohort splits that need DC run on 2023-2025; the raw
HC-only split runs on the full window.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.coaching_changes import coaching_change_table  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE")
PRIOR_GAMES_GRID = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 55]


def _weekly_pos_rows(year):
    df, _tcol, _ncol, _ = load_and_merge_data(year, "Full PPR")
    df = df[["week", "position", "opponent_team", "fantasy_points_ppr"]].copy()
    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(POSITIONS) & df["opponent_team"].notna() & df["week"].notna()]
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["fantasy_points_ppr"] = pd.to_numeric(df["fantasy_points_ppr"], errors="coerce").fillna(0.0)
    df = df[(df["week"] >= 1) & (df["week"] <= 18)]
    return df


def _defense_rating(rows, week_lo=1, week_hi=18):
    """Per-(defense team, position) allowed-PPR-per-game / league mean, in the
    given week window. Returns a wide frame indexed by team, columns = POSITIONS."""
    w = rows[(rows["week"] >= week_lo) & (rows["week"] <= week_hi)]
    if w.empty:
        return pd.DataFrame(columns=POSITIONS)
    per_team_game = (w.groupby(["opponent_team", "position", "week"])["fantasy_points_ppr"]
                     .sum().reset_index())
    allowed = (per_team_game.groupby(["opponent_team", "position"])["fantasy_points_ppr"]
               .mean().unstack("position"))
    league = allowed.mean(axis=0)
    rating = allowed.divide(league, axis=1)
    return rating.reindex(columns=POSITIONS)


def year_over_year_persistence(years, cohorts):
    coach = coaching_change_table(min(years) - 1, max(years))
    ratings = {y: _defense_rating(_weekly_pos_rows(y)) for y in range(min(years) - 1, max(years) + 1)}

    recs = []
    for y in years:
        cur, prev = ratings.get(y), ratings.get(y - 1)
        if cur is None or prev is None or cur.empty or prev.empty:
            continue
        cc = coach[coach["season"] == y].set_index("team")
        for team in cur.index.intersection(prev.index):
            if team not in cc.index:
                continue
            row = cc.loc[team]
            for pos in POSITIONS:
                if np.isfinite(cur.loc[team, pos]) and np.isfinite(prev.loc[team, pos]):
                    recs.append({
                        "season": y, "team": team, "pos": pos,
                        "prev_rating": float(prev.loc[team, pos]),
                        "cur_rating": float(cur.loc[team, pos]),
                        "hc_changed": row["hc_changed"], "dc_changed": row.get("dc_changed"),
                        "cohort": row["coaching_cohort"],
                    })
    d = pd.DataFrame(recs)
    if d.empty:
        print("no persistence rows")
        return d

    def _report(sub, label):
        if len(sub) < 12:
            print(f"  {label:<26} n={len(sub):<4} (too few)")
            return
        r = np.corrcoef(sub["prev_rating"], sub["cur_rating"])[0, 1]
        mae = (sub["cur_rating"] - sub["prev_rating"]).abs().mean()
        yoy_sd = (sub["cur_rating"] - sub["prev_rating"]).std()
        print(f"  {label:<26} n={len(sub):<4} corr(prev,cur)={r:+.3f}  "
              f"|d-rating| MAE={mae:.3f}  sd(Δ)={yoy_sd:.3f}")

    print("\n=== 1. YEAR-OVER-YEAR DEFENSE-ALLOWED PERSISTENCE ===")
    print(f"window {years[0]}-{years[-1]}, {d['team'].nunique()} teams, {len(d)} team-pos rows\n")
    print("By head-coach change (full window):")
    _report(d[d["hc_changed"] == False], "HC unchanged")   # noqa: E712
    _report(d[d["hc_changed"] == True], "HC changed")       # noqa: E712
    dc = d[d["dc_changed"].notna()]
    if not dc.empty:
        print(f"\nBy DC change (Ourlads window, {dc['season'].min():.0f}-{dc['season'].max():.0f}):")
        _report(dc[dc["dc_changed"] == False], "DC unchanged")   # noqa: E712
        _report(dc[dc["dc_changed"] == True], "DC changed")      # noqa: E712
        print("\nBy 4-way cohort:")
        for c in ("none", "dc_only", "hc_only", "both", "dc_to_hc"):
            _report(dc[dc["cohort"] == c], c)
        print("\nDC change, per position:")
        for pos in POSITIONS:
            pp = dc[dc["pos"] == pos]
            _report(pp[pp["dc_changed"] == False], f"{pos} DC unchanged")  # noqa: E712
            _report(pp[pp["dc_changed"] == True], f"{pos} DC changed")     # noqa: E712
    return d


def _fmt_grid_row(label, by_pg, min_obs=200):
    nobs = len(by_pg[PRIOR_GAMES_GRID[0]])
    if nobs < min_obs:
        print(f"{label:<20}{nobs:>7}  (too few)")
        return None
    maes = {pg: float(np.mean(v)) for pg, v in by_pg.items()}
    best = min(maes, key=maes.get)
    cells = "".join((f"*{maes[pg]:.3f}" if pg == best else f" {maes[pg]:.3f}")
                    for pg in PRIOR_GAMES_GRID)
    print(f"{label:<20}{nobs:>7}  {cells}  best={best}")
    return best


def optimal_prior_weight(years):
    coach = coaching_change_table(min(years) - 1, max(years))
    rows_by_year = {y: _weekly_pos_rows(y) for y in range(min(years) - 1, max(years) + 1)}
    prev_full = {y: _defense_rating(rows_by_year[y]) for y in rows_by_year}

    def _blank():
        return {pg: [] for pg in PRIOR_GAMES_GRID}

    by_cohort = {c: _blank() for c in ("all", "none", "dc_only", "hc_only", "both", "dc_to_hc",
                                       "hc_unchanged", "hc_changed")}
    by_cohort_pos = {}
    by_cohort_year = {}
    for y in years:
        cur_rows = rows_by_year.get(y)
        prev = prev_full.get(y - 1)
        if cur_rows is None or prev is None or prev.empty:
            continue
        cc = coach[coach["season"] == y].set_index("team")
        for n in range(2, 15):
            to_date = _defense_rating(cur_rows, 1, n)
            future = _defense_rating(cur_rows, n + 1, 18)
            teams = to_date.index.intersection(future.index).intersection(prev.index)
            for team in teams:
                if team not in cc.index:
                    continue
                cohort = cc.loc[team, "coaching_cohort"]
                hc_c = cc.loc[team, "hc_changed"]
                for pos in POSITIONS:
                    cd, fu, pr = to_date.loc[team, pos], future.loc[team, pos], prev.loc[team, pos]
                    if not (np.isfinite(cd) and np.isfinite(fu) and np.isfinite(pr)):
                        continue
                    for pg in PRIOR_GAMES_GRID:
                        a = n / (n + pg)
                        err = abs(a * cd + (1 - a) * pr - fu)
                        by_cohort["all"][pg].append(err)
                        if cohort in by_cohort:
                            by_cohort[cohort][pg].append(err)
                            by_cohort_pos.setdefault((cohort, pos), _blank())[pg].append(err)
                            by_cohort_year.setdefault((cohort, y), _blank())[pg].append(err)
                        if hc_c is True:
                            by_cohort["hc_changed"][pg].append(err)
                        elif hc_c is False:
                            by_cohort["hc_unchanged"][pg].append(err)

    hdr = f"{'group':<20}{'nobs':>7}  " + "".join(f"pg{pg:>4}" for pg in PRIOR_GAMES_GRID)
    print("\n\n=== 2. OPTIMAL prior_games BY COHORT (out-of-sample within-season) ===")
    print(f"window {years[0]}-{years[-1]}; blend weeks-1..n vs actual weeks-(n+1)..18, n=2..14\n")
    print(hdr)
    for c, by_pg in by_cohort.items():
        _fmt_grid_row(c, by_pg)

    print("\n-- by cohort x position --")
    print(hdr)
    for c in ("none", "dc_only", "both"):
        for pos in POSITIONS:
            key = (c, pos)
            if key in by_cohort_pos:
                _fmt_grid_row(f"{c}/{pos}", by_cohort_pos[key], min_obs=120)

    print("\n-- by cohort x season (stability check) --")
    print(hdr)
    for c in ("none", "dc_only", "both"):
        for y in years:
            key = (c, y)
            if key in by_cohort_year:
                _fmt_grid_row(f"{c}/{y}", by_cohort_year[key], min_obs=80)

    print("\n(* = min MAE in the row. Larger best-pg => prior season trusted longer.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2018,2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--dc-years", default="2023,2024,2025",
                    help="restrict the DC-cohort analyses to seasons with Ourlads data")
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(",")]
    dc_years = [int(y) for y in args.dc_years.split(",")]

    year_over_year_persistence(years, None)
    print("\n" + "-" * 78)
    year_over_year_persistence(dc_years, None)   # DC-focused re-run on its own window
    print("\n" + "-" * 78)
    optimal_prior_weight(years)
    print("\n" + "-" * 78)
    optimal_prior_weight(dc_years)


if __name__ == "__main__":
    main()
