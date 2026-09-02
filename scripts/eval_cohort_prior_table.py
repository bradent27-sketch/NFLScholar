"""Fast matchup-accuracy readout for the position x cohort DEFENSE_PRIOR_GAMES
table - isolates the mechanism from the full-projection noise.

Target = fantasy-points-allowed per game to a position, as a rating vs the
season's cross-defense mean (this is what the model's matchup_matrix scales).
For a grid of "spread strength" lambda, interpolate every cell of
data.coaching_changes._POS_COHORT_DEFAULTS between the shipped scalar (12) and
its fitted value, then, strictly out of sample within season:

    blended = a * (weeks 1..n allowed rating) + (1-a) * (prior-year rating),
    a = n / (n + pg[pos, cohort])

score |blended - actual weeks (n+1)..18 rating|, n = 2..8. Report the table's
pooled + per-(pos, cohort) MAE minus the flat pg=12 baseline's. Negative =
the coaching-aware table beats flat-12 on the matchup input itself.

    python scripts/eval_cohort_prior_table.py --years 2020-2025
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

from data.coaching_changes import coaching_change_table, _POS_COHORT_DEFAULTS  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE")
COHORTS = ("none", "dc_only", "both", "hc_only")
FLAT = 12.0
N_RANGE = range(2, 9)          # week cutoffs where the prior still carries weight
CLAMP = (4.0, 30.0)

LAMBDAS = [("flat_l0.00", 0.00, "all"), ("l0.35", 0.35, "all"), ("l0.60", 0.60, "all"),
           ("l0.85", 0.85, "all"), ("l1.00", 1.00, "all"), ("l1.25", 1.25, "all"),
           ("both_only_l1", 1.00, "reset"), ("stable_only_l1", 1.00, "stable")]
_RESET_COH = {"both", "hc_only"}


def _pg_table(lam, shape):
    """{pos: {cohort: pg}} at spread strength ``lam``. ``shape`` limits which
    cohorts move: 'all', 'reset' (both/hc_only only), 'stable' (none/dc_only)."""
    out = {}
    for pos, d in _POS_COHORT_DEFAULTS.items():
        out[pos] = {}
        for coh, fitted in d.items():
            if coh not in COHORTS:
                out[pos][coh] = FLAT
                continue
            move = (shape == "all"
                    or (shape == "reset" and coh in _RESET_COH)
                    or (shape == "stable" and coh not in _RESET_COH))
            if fitted is None or not move:
                out[pos][coh] = FLAT
            else:
                out[pos][coh] = float(np.clip(FLAT + lam * (fitted - FLAT), *CLAMP))
    return out


def _load(years):
    ps = nfl.load_player_stats(seasons=list(years)).to_pandas()
    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]
    col = "fantasy_points_ppr" if "fantasy_points_ppr" in ps.columns else "fantasy_points"
    ps = ps[["position", "season", "week", "team", col]].copy()
    ps["position"] = ps["position"].astype(str).str.upper()
    ps = ps[ps["position"].isin(POSITIONS)]
    ps["week"] = pd.to_numeric(ps["week"], errors="coerce")
    ps[col] = pd.to_numeric(ps[col], errors="coerce").fillna(0.0)
    ps = ps[(ps["week"] >= 1) & (ps["week"] <= 18)].rename(columns={col: "fp"})

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


def _rating(df, pos, wlo, whi):
    d = df[(df["position"] == pos) & (df["week"] >= wlo) & (df["week"] <= whi)]
    if d.empty:
        return pd.Series(dtype=float)
    per_game = d.groupby(["def_team", "week"])["fp"].sum().groupby("def_team").mean()
    lg = per_game.mean()
    return per_game / lg if lg else per_game * np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))

    df = _load(range(lo - 1, hi + 1))
    coh = {(int(r.season), r.team): r.coaching_cohort
           for r in coaching_change_table(lo - 1, hi).itertuples(index=False)}
    by_year = {y: df[df["season"] == y] for y in range(lo - 1, hi + 1)}

    # collect every out-of-sample (pos, cohort, year, n, team) sample once:
    # (to_date_rating, future_rating, prior_rating)
    samples = []
    for y in years:
        cur, prev_df = by_year.get(y), by_year.get(y - 1)
        if cur is None or prev_df is None or prev_df.empty:
            continue
        for pos in POSITIONS:
            prev = _rating(prev_df, pos, 1, 18)
            if prev.empty:
                continue
            for n in N_RANGE:
                td = _rating(cur, pos, 1, n)
                fu = _rating(cur, pos, n + 1, 18)
                for tm in td.index.intersection(fu.index).intersection(prev.index):
                    c0, f0, p0 = td.get(tm), fu.get(tm), prev.get(tm)
                    if np.isfinite(c0) and np.isfinite(f0) and np.isfinite(p0):
                        c = coh.get((y, tm), "unknown")
                        samples.append((pos, c, n, c0, f0, p0))
    S = pd.DataFrame(samples, columns=["pos", "cohort", "n", "td", "fu", "prior"])
    print(f"out-of-sample matchup samples {years[0]}-{years[-1]}: {len(S):,}")
    print("by cohort:", dict(S["cohort"].value_counts()))

    def _mae(pg_of):
        a = S["n"] / (S["n"] + S.apply(lambda r: pg_of(r["pos"], r["cohort"]), axis=1))
        blend = a * S["td"] + (1 - a) * S["prior"]
        return (blend - S["fu"]).abs()

    base_err = _mae(lambda p, c: FLAT)
    base_by = base_err.groupby([S["pos"], S["cohort"]]).mean()
    base_pool = base_err.mean()

    print(f"\nflat pg=12 baseline matchup-rating MAE: {base_pool:.4f}\n")
    hdr = f"{'config':<16}{'pooled dMAE':>13}   " + "".join(f"{c:>11}" for c in COHORTS)
    print(hdr + "\n" + "-" * len(hdr))
    for label, lam, shape in LAMBDAS:
        tbl = _pg_table(lam, shape)
        err = _mae(lambda p, c: tbl.get(p, {}).get(c, FLAT))
        d_pool = err.mean() - base_pool
        by = err.groupby([S["pos"], S["cohort"]]).mean() - base_by
        # cohort-level dMAE, averaged across the 4 positions
        cells = []
        for c in COHORTS:
            vals = [by.get((p, c), np.nan) for p in POSITIONS]
            cells.append(f"{np.nanmean(vals):+.4f}")
        star = "  <-- control (expect ~0)" if lam == 0 else ""
        print(f"{label:<16}{d_pool:>+13.4f}   " + "".join(f"{v:>11}" for v in cells) + star)

    print("\n-- per (pos x cohort) dMAE vs flat-12, at l1.00 (shape=all) --")
    tbl = _pg_table(1.0, "all")
    err = _mae(lambda p, c: tbl.get(p, {}).get(c, FLAT))
    by = (err.groupby([S["pos"], S["cohort"]]).mean() - base_by).unstack("cohort")
    print(by.reindex(columns=[c for c in COHORTS if c in by.columns]).round(4).to_string())


if __name__ == "__main__":
    main()
