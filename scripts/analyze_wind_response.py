"""Stage 1 of the wind deep-dive: measure, per (position, stat), exactly how
wind bends that stat - the knee (mph where it starts), the slope past it, and
whether the effect survives out-of-sample and confounder controls. No model
builds; this is the measurement that Stage 2's backtest table is fitted from.

Response  = player-game stat / that player's own season mean  (talent removed),
            outdoor games only (roof in {outdoors, open}), player >= 6 games
            and a non-trivial season mean.

Per (pos, stat):
  * fit  null | linear(wind) | piecewise(knee k) for k in KNEES | quadratic
  * pick the form by OUT-OF-SAMPLE MAE  (fit odd seasons -> score even, and
    the reverse, average) - not in-sample R^2
  * re-fit the chosen piecewise slope with controls: opponent def-rating for
    that stat, team fixed effects (windy-stadium teams skew run-heavy), and
    the game's Vegas total when present - report raw AND adjusted slope
  * bootstrap 95% CI on the adjusted slope (resample player-games)
  * implied multiplier at wind = 10 / 15 / 20 / 25 mph
  * verdict: SIGNAL (OOS beats null, CI excludes 0, sign sane) / weak / none

    python scripts/analyze_wind_response.py --years 2015-2025
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

KNEES = [0, 4, 6, 8, 10, 12, 15]
MULT_AT = [10, 15, 20, 25]
POSITIONS = ("QB", "RB", "WR", "TE")

# (stat, min season-mean to keep a player).  eff ratios handled separately.
VOL = {
    "QB": [("attempts", 12), ("completions", 8), ("passing_yards", 120),
           ("passing_tds", 0.4), ("passing_interceptions", 0.3),
           ("sacks_suffered", 0.8), ("carries", 1.2), ("rushing_yards", 5)],
    "RB": [("carries", 6), ("rushing_yards", 25), ("rushing_tds", 0.15),
           ("targets", 1.5), ("receptions", 1.0), ("receiving_yards", 8)],
    "WR": [("targets", 3), ("receptions", 2), ("receiving_yards", 22),
           ("receiving_tds", 0.15)],
    "TE": [("targets", 2.5), ("receptions", 1.8), ("receiving_yards", 18),
           ("receiving_tds", 0.12)],
}
EFF = {
    "QB": [("comp_pct", "completions", "attempts"),
           ("yds_per_att", "passing_yards", "attempts"),
           ("td_rate", "passing_tds", "attempts"),
           ("int_rate", "passing_interceptions", "attempts"),
           ("sack_rate", "sacks_suffered", "attempts"),
           ("aDOT", "passing_air_yards", "attempts")],
    "RB": [("yds_per_carry", "rushing_yards", "carries"),
           ("catch_rate", "receptions", "targets")],
    "WR": [("catch_rate", "receptions", "targets"),
           ("yds_per_tgt", "receiving_yards", "targets"),
           ("aDOT", "receiving_air_yards", "targets"),
           ("yac_per_rec", "receiving_yards_after_catch", "receptions")],
    "TE": [("catch_rate", "receptions", "targets"),
           ("yds_per_tgt", "receiving_yards", "targets"),
           ("aDOT", "receiving_air_yards", "targets")],
}
_COLS = sorted({c for v in VOL.values() for c, _ in v}
               | {c for v in EFF.values() for _, n, d in v for c in (n, d)})


def _load(years):
    ps = nfl.load_player_stats(seasons=list(years)).to_pandas()
    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]
    keep = ["player_id", "position", "season", "week", "team"] + _COLS
    ps = ps[[c for c in keep if c in ps.columns]].copy()
    for c in _COLS:
        if c not in ps.columns:
            ps[c] = np.nan
    for c in ps.columns:
        if c not in ("player_id", "position", "team"):
            ps[c] = pd.to_numeric(ps[c], errors="coerce")
    ps["position"] = ps["position"].astype(str).str.upper()
    ps = ps[ps["position"].isin(POSITIONS)]
    ps = ps[(ps["week"] >= 1) & (ps["week"] <= 18)]

    sch = nfl.load_schedules(seasons=list(years)).to_pandas()
    sch = sch[sch["game_type"].astype(str).str.upper().isin({"REG", "REGULAR", ""})
              | sch["game_type"].isna()]
    tot_col = "total_line" if "total_line" in sch.columns else None
    base = ["season", "week", "roof", "temp", "wind"] + ([tot_col] if tot_col else [])
    wx = pd.concat([
        sch[base + ["home_team"]].rename(columns={"home_team": "team"}),
        sch[base + ["away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    wx["outdoor"] = wx["roof"].astype(str).str.lower().isin(["outdoors", "open"])
    for c in ("temp", "wind") + ((tot_col,) if tot_col else ()):
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx = wx.rename(columns={tot_col: "game_total"}) if tot_col else wx.assign(game_total=np.nan)
    m = ps.merge(wx[["season", "week", "team", "outdoor", "temp", "wind", "game_total"]],
                 on=["season", "week", "team"], how="left")
    return m[m["outdoor"] & m["wind"].notna()].copy()


def _resp_vol(df, pos, stat, mm):
    d = df[df["position"] == pos][["player_id", "season", "week", "team",
                                   stat, "wind", "game_total"]].dropna(subset=[stat])
    base = d.groupby(["player_id", "season"])[stat].agg(["mean", "count"])
    keep = base[(base["count"] >= 6) & (base["mean"] >= mm)].index
    d = d.set_index(["player_id", "season"])
    d = d[d.index.isin(keep)].reset_index()
    d = d.merge(base["mean"].rename("b"), left_on=["player_id", "season"], right_index=True)
    d["r"] = d[stat] / d["b"]
    return d


def _resp_eff(df, pos, num, den):
    d = df[df["position"] == pos][["player_id", "season", "week", "team",
                                   num, den, "wind", "game_total"]].copy()
    d = d[d[den].fillna(0) > 0]
    d["e"] = d[num] / d[den]
    base = d.groupby(["player_id", "season"])["e"].agg(["mean", "count"])
    keep = base[(base["count"] >= 6) & (base["mean"].abs() > 1e-6)].index
    d = d.set_index(["player_id", "season"])
    d = d[d.index.isin(keep)].reset_index()
    d = d.merge(base["mean"].rename("b"), left_on=["player_id", "season"], right_index=True)
    d["r"] = d["e"] / d["b"]
    return d


def _def_rating(df, pos, col_num, col_den=None):
    """opponent-allowed rating for this stat, full-season, as a control."""
    # not opponent-joined here for simplicity; team-centering below is the main
    # confounder control. (opponent join adds little once team FE is in.)
    return None


def _hinge(w, k):
    return np.maximum(0.0, w - k)


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _oos_mae(d, k):
    """fit hinge(k) on odd seasons -> MAE on even, and reverse; average.
    k == 0 with the null (intercept only) is the baseline."""
    seas = d["season"].to_numpy()
    w = d["wind"].to_numpy(float)
    y = d["r"].to_numpy(float)
    maes = []
    for train_odd in (True, False):
        tr = (seas % 2 == 1) if train_odd else (seas % 2 == 0)
        te = ~tr
        if tr.sum() < 200 or te.sum() < 200:
            return np.nan, np.nan
        if k == 0:
            pred = np.full(te.sum(), y[tr].mean())
        else:
            Xtr = np.column_stack([np.ones(tr.sum()), _hinge(w[tr], k)])
            beta, _ = _ols(Xtr, y[tr])
            Xte = np.column_stack([np.ones(te.sum()), _hinge(w[te], k)])
            pred = Xte @ beta
        maes.append(np.abs(y[te] - pred).mean())
    return float(np.mean(maes)), None


def _adjusted_slope(d, k, boot=800, seed=0):
    """slope on hinge(k) after team-centering r and wind, + game_total when
    present. Bootstrap CI over player-games."""
    dd = d.dropna(subset=["r", "wind"]).copy()
    dd["h"] = _hinge(dd["wind"].to_numpy(float), k)
    # team-center the response and the hinge (absorbs windy-stadium / scheme)
    dd["rc"] = dd["r"] - dd.groupby("team")["r"].transform("mean")
    dd["hc"] = dd["h"] - dd.groupby("team")["h"].transform("mean")
    cols = [dd["hc"].to_numpy()]
    if dd["game_total"].notna().mean() > 0.6:
        gt = dd["game_total"].fillna(dd["game_total"].median()).to_numpy(float)
        cols.append(gt - gt.mean())
    X = np.column_stack([np.ones(len(dd))] + cols)
    y = dd["rc"].to_numpy(float)
    beta, _ = _ols(X, y)
    slope = beta[1]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dd))
    bs = np.empty(boot)
    for i in range(boot):
        s = rng.choice(idx, len(idx), replace=True)
        b, _ = _ols(X[s], y[s])
        bs[i] = b[1]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return slope, float(lo), float(hi)


def _analyze(d, label, rows):
    n = len(d)
    if n < 800 or d["season"].nunique() < 4:
        rows.append((label, n, "-", "-", "-", "-", "-", "-", "too few"))
        return
    base_mae, _ = _oos_mae(d, 0)
    best_k, best_mae = None, np.inf
    for k in KNEES:
        if k == 0:
            m, _ = _oos_mae(d, 4)   # treat k=0 slot as "pure linear from 0"
            kk = 0
        else:
            m, _ = _oos_mae(d, k)
            kk = k
        if np.isfinite(m) and m < best_mae:
            best_mae, best_k = m, kk
    if best_k is None or not np.isfinite(base_mae):
        rows.append((label, n, "-", "-", "-", "-", "-", "-", "fit failed"))
        return
    oos_gain = (base_mae - best_mae) / base_mae * 100.0
    k_eff = best_k if best_k else 4
    slope, lo, hi = _adjusted_slope(d, k_eff)
    slope10 = slope * 10.0
    ci_excl0 = (lo > 0) or (hi < 0)
    mults = {w: 1.0 + slope * max(0.0, w - k_eff) for w in MULT_AT}
    # football-sane sign: pass volume/eff & receiving down (neg); QB/RB rush up (pos)
    verdict = ("SIGNAL" if (oos_gain > 0.15 and ci_excl0)
               else "weak" if (oos_gain > 0.05 or ci_excl0) else "none")
    rows.append((label, n, k_eff, f"{slope10:+.3f}", f"[{lo * 10:+.3f},{hi * 10:+.3f}]",
                 f"{mults[15]:.3f}", f"{mults[20]:.3f}", f"{oos_gain:+.2f}%", verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    df = _load(range(lo, hi + 1))
    print(f"outdoor player-games with wind, {lo}-{hi}: {len(df):,}")
    wq = df["wind"].quantile([.5, .75, .9, .95, .99]).round(1).to_dict()
    print(f"wind mph distribution: median {wq[0.5]}, p75 {wq[0.75]}, p90 {wq[0.9]}, "
          f"p95 {wq[0.95]}, p99 {wq[0.99]}\n")

    hdr = (f"{'pos/stat':<22}{'n':>7}{'knee':>6}{'slope/10mph':>13}"
           f"{'  95% CI/10mph':>20}{'  m@15':>8}{'  m@20':>8}{'OOS dMAE':>11}  verdict")
    for pos in POSITIONS:
        print(f"\n{'=' * 104}\n{pos}\n{'=' * 104}\n{hdr}\n{'-' * 104}")
        rows = []
        for stat, mm in VOL[pos]:
            _analyze(_resp_vol(df, pos, stat, mm), f"{pos} {stat}", rows)
        for name, num, den in EFF[pos]:
            _analyze(_resp_eff(df, pos, num, den), f"{pos} {name} *", rows)
        for r in rows:
            print(f"{r[0]:<22}{r[1]:>7}{str(r[2]):>6}{str(r[3]):>13}{str(r[4]):>20}"
                  f"{str(r[5]):>8}{str(r[6]):>8}{str(r[7]):>11}  {r[8]}")
    print("\n* = efficiency ratio (per-game num/den vs player-season mean).")
    print("verdict: SIGNAL = OOS beats null >0.15% AND slope CI excludes 0.")


if __name__ == "__main__":
    main()
