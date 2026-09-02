"""Per-STAT defense-allowed persistence by coaching cohort.

The whole-pool fantasy-PPG view (scripts/analyze_coaching_defense_prior.py)
found the coaching-aware prior effect real but tiny. The user's point: a fantasy
PPG is a blunt aggregate. A defense can lose its edge against WR deep balls
while keeping it against TE seams; rush-yards-allowed is personnel-driven and
may survive a coordinator swap that wrecks the pass-rush/coverage profile.
If some STAT resets hard on a DC change while another persists, THAT stat is
where a coaching-aware prior_games could actually pay - even though the blended
PPG washes it out.

This script, per coaching cohort (none / dc_only / hc_only / both / dc_to_hc;
see data.coaching_changes) and per projected stat:

  1. PERSISTENCE: defense-allowed rating_{Y-1} vs rating_Y, Pearson corr +
     mean |Y-o-Y move|. A stat whose corr holds up across a DC change does
     not need a shorter leash; one that collapses does.
  2. OPTIMAL prior_games, out of sample within season: blend the team's
     weeks-1..n allowed rating with its prior-year rating (weight n/(n+pg))
     and score against its ACTUAL weeks-(n+1)..18 rating, for a grid of pg.
     Best pg per (stat, cohort). If `none` wants pg=20 for WR TDs but `both`
     wants pg=6, the model should make DEFENSE_PRIOR_GAMES stat-and-cohort
     aware for that stat.

Rating = (stat allowed to that position per game) / (that season's league mean
allowed to the position), so ~1.0 = average defense, <1 = stingy. Efficiency
ratios (comp%, yds/att, catch rate, aDOT, YAC/rec) are season-sum(num) /
season-sum(den), then / league mean.

Alignment note: nflverse weekly player_stats has no wide/slot split, so a true
"guards the boundary WR but not the slot" cut needs FTN/PFF charting this
pipeline does not load. aDOT-allowed and YAC/reception-allowed are the
scheme proxies available here (deep-zone vs press-man leaves a fingerprint in
both).

    python scripts/analyze_coaching_defense_prior_bystat.py --years 2020-2025
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import nflreadpy as nfl  # noqa: E402

from data.coaching_changes import coaching_change_table  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE")
PRIOR_GAMES_GRID = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 55]
COHORTS = ("all", "none", "dc_only", "hc_only", "both", "dc_to_hc")

# volume stats to sum per (defense, position, week): model-projected counting
# stats only.
VOL = {
    "QB": ["attempts", "completions", "passing_yards", "passing_tds",
           "passing_interceptions", "sacks_suffered", "carries", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions",
           "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds"],
}
# efficiency ratios: (label, numerator, denominator) - summed over the window
# then ratio'd, so they are exposure-weighted, not a mean of per-game ratios.
EFF = {
    "QB": [("comp_pct", "completions", "attempts"),
           ("yds_per_att", "passing_yards", "attempts"),
           ("td_rate", "passing_tds", "attempts"),
           ("int_rate", "passing_interceptions", "attempts"),
           ("sack_rate", "sacks_suffered", "attempts"),
           ("aDOT", "passing_air_yards", "attempts")],
    "RB": [("yds_per_carry", "rushing_yards", "carries"),
           ("catch_rate", "receptions", "targets"),
           ("yds_per_tgt", "receiving_yards", "targets")],
    "WR": [("catch_rate", "receptions", "targets"),
           ("yds_per_tgt", "receiving_yards", "targets"),
           ("td_per_tgt", "receiving_tds", "targets"),
           ("aDOT", "receiving_air_yards", "targets"),
           ("yac_per_rec", "receiving_yards_after_catch", "receptions")],
    "TE": [("catch_rate", "receptions", "targets"),
           ("yds_per_tgt", "receiving_yards", "targets"),
           ("td_per_tgt", "receiving_tds", "targets"),
           ("aDOT", "receiving_air_yards", "targets"),
           ("yac_per_rec", "receiving_yards_after_catch", "receptions")],
}
_ALL_COLS = sorted({c for v in VOL.values() for c in v}
                   | {c for v in EFF.values() for _, n, d in v for c in (n, d)})


def _load(years):
    ps = nfl.load_player_stats(seasons=list(years)).to_pandas()
    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]
    keep = ["player_id", "position", "season", "week", "team"] + _ALL_COLS
    ps = ps[[c for c in keep if c in ps.columns]].copy()
    for c in ps.columns:
        if c not in ("player_id", "position", "team"):
            ps[c] = pd.to_numeric(ps[c], errors="coerce")
    for c in _ALL_COLS:                       # missing cols (older seasons) -> 0
        if c not in ps.columns:
            ps[c] = 0.0
    ps["position"] = ps["position"].astype(str).str.upper()
    ps = ps[ps["position"].isin(POSITIONS)]
    ps["week"] = pd.to_numeric(ps["week"], errors="coerce")
    ps = ps[(ps["week"] >= 1) & (ps["week"] <= 18)]

    sch = nfl.load_schedules(seasons=list(years)).to_pandas()
    sch = sch[sch["game_type"].astype(str).str.upper().isin({"REG", "REGULAR", ""})
              | sch["game_type"].isna()]
    opp = pd.concat([
        sch[["season", "week", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "def_team"}),
        sch[["season", "week", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "def_team"}),
    ], ignore_index=True)
    opp["week"] = pd.to_numeric(opp["week"], errors="coerce")
    m = ps.merge(opp, on=["season", "week", "team"], how="inner")
    m["def_team"] = m["def_team"].astype(str).str.upper().replace(
        {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"})
    return m


def _rating(df, pos, stat, wlo=1, whi=18):
    """defense-allowed per game for `stat` to `pos`, over weeks [wlo, whi],
    divided by that slice's cross-defense mean. Series indexed by def_team."""
    d = df[(df["position"] == pos) & (df["week"] >= wlo) & (df["week"] <= whi)]
    if d.empty:
        return pd.Series(dtype=float)
    per_game = (d.groupby(["def_team", "week"])[stat].sum()
                .groupby("def_team").mean())
    lg = per_game.mean()
    return per_game / lg if lg else per_game * np.nan


def _eff_rating(df, pos, num, den, wlo=1, whi=18):
    d = df[(df["position"] == pos) & (df["week"] >= wlo) & (df["week"] <= whi)]
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby("def_team")[[num, den]].sum()
    r = g[num] / g[den].where(g[den] > 0)
    lg = r.mean()
    return r / lg if lg else r * np.nan


def _series_for(df, pos, kind, spec, wlo=1, whi=18):
    if kind == "vol":
        return _rating(df, pos, spec, wlo, whi)
    return _eff_rating(df, pos, spec[1], spec[2], wlo, whi)


def _cohort_lookup(years):
    tbl = coaching_change_table(min(years) - 1, max(years))
    return {(int(r.season), r.team): r.coaching_cohort
            for r in tbl.itertuples(index=False)}


def persistence(df, years, cohort_of):
    print(f"\n{'=' * 100}\n1. YEAR-OVER-YEAR DEFENSE-ALLOWED PERSISTENCE BY STAT x COHORT\n"
          f"   window {years[0]}-{years[-1]};  corr(prev, cur) / mean|delta|  (n team-seasons)\n{'=' * 100}")
    ratings = {y: df[df["season"] == y] for y in range(min(years) - 1, max(years) + 1)}

    for pos in POSITIONS:
        items = [("vol", s, s) for s in VOL[pos]] + [("eff", lbl, (lbl, n, d))
                                                     for (lbl, n, d) in EFF[pos]]
        print(f"\n--- {pos} ---")
        print(f"{'stat':<14}" + "".join(f"{c:>16}" for c in COHORTS))
        for kind, label, spec in items:
            recs = {c: [] for c in COHORTS}
            for y in years:
                cur_df, prev_df = ratings.get(y), ratings.get(y - 1)
                if cur_df is None or prev_df is None or cur_df.empty or prev_df.empty:
                    continue
                cur = _series_for(cur_df, pos, kind, spec)
                prev = _series_for(prev_df, pos, kind, spec)
                common = cur.index.intersection(prev.index)
                for tm in common:
                    cv, pv = cur.get(tm), prev.get(tm)
                    if not (np.isfinite(cv) and np.isfinite(pv)):
                        continue
                    coh = cohort_of.get((y, tm), "unknown")
                    recs["all"].append((pv, cv))
                    if coh in recs:
                        recs[coh].append((pv, cv))
            cells = []
            for c in COHORTS:
                pairs = recs[c]
                if len(pairs) < 12:
                    cells.append(f"{'-':>16}")
                    continue
                a = np.array(pairs)
                r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
                md = np.abs(a[:, 1] - a[:, 0]).mean()
                cells.append(f"{r:+.2f}/{md:.2f}(n{len(pairs)})".rjust(16))
            print(f"{label:<14}" + "".join(cells))


def optimal_prior_games(df, years, cohort_of):
    print(f"\n\n{'=' * 100}\n2. OPTIMAL prior_games BY STAT x COHORT  (out-of-sample within season)\n"
          f"   blend weeks-1..n with prior year, weight n/(n+pg); score vs actual weeks-(n+1)..18, n=2..14\n"
          f"   window {years[0]}-{years[-1]};  '*' marks min-MAE pg in the row\n{'=' * 100}")
    by_year = {y: df[df["season"] == y] for y in range(min(years) - 1, max(years) + 1)}
    hdr = f"{'stat':<14}{'cohort':<10}{'nobs':>7}  " + "".join(f"{('pg' + str(pg)):>7}" for pg in PRIOR_GAMES_GRID) + "   best"

    for pos in POSITIONS:
        items = [("vol", s, s) for s in VOL[pos]] + [("eff", lbl, (lbl, n, d))
                                                     for (lbl, n, d) in EFF[pos]]
        print(f"\n--- {pos} ---\n{hdr}")
        for kind, label, spec in items:
            acc = {c: {pg: [] for pg in PRIOR_GAMES_GRID} for c in COHORTS}
            for y in years:
                cur_df, prev_df = by_year.get(y), by_year.get(y - 1)
                if cur_df is None or prev_df is None or prev_df.empty:
                    continue
                prev = _series_for(prev_df, pos, kind, spec)
                if prev.empty:
                    continue
                for n in range(2, 15):
                    td = _series_for(cur_df, pos, kind, spec, 1, n)
                    fu = _series_for(cur_df, pos, kind, spec, n + 1, 18)
                    common = td.index.intersection(fu.index).intersection(prev.index)
                    for tm in common:
                        c0, f0, p0 = td.get(tm), fu.get(tm), prev.get(tm)
                        if not (np.isfinite(c0) and np.isfinite(f0) and np.isfinite(p0)):
                            continue
                        coh = cohort_of.get((y, tm), "unknown")
                        for pg in PRIOR_GAMES_GRID:
                            a = n / (n + pg)
                            err = abs(a * c0 + (1 - a) * p0 - f0)
                            acc["all"][pg].append(err)
                            if coh in acc:
                                acc[coh][pg].append(err)
            for c in COHORTS:
                nobs = len(acc[c][PRIOR_GAMES_GRID[0]])
                if nobs < 150:
                    print(f"{label:<14}{c:<10}{nobs:>7}  (too few)")
                    continue
                maes = {pg: float(np.mean(acc[c][pg])) for pg in PRIOR_GAMES_GRID}
                best = min(maes, key=maes.get)
                cells = "".join((f"*{maes[pg]:.3f}" if pg == best else f" {maes[pg]:.3f}")
                                for pg in PRIOR_GAMES_GRID)
                print(f"{label:<14}{c:<10}{nobs:>7}  {cells}   {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))

    df = _load(range(lo - 1, hi + 1))
    print(f"player-games with an opponent, {lo - 1}-{hi}: {len(df):,}")
    cohort_of = _cohort_lookup(years)
    coh_counts = pd.Series(list(cohort_of.values())).value_counts()
    print("cohort counts in window (defense team-seasons):")
    print(coh_counts.to_string())

    persistence(df, years, cohort_of)
    optimal_prior_games(df, years, cohort_of)


if __name__ == "__main__":
    main()
