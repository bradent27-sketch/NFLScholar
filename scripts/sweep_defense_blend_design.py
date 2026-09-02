"""Phase-1 design-space sweep for the defense matchup blend, scored PER STAT
per position per season-phase on the defense-allowed-rating proxy (fast, no
model builds). Answers: how sharply should last season's defense profile fade
as this season's sample fills in, does that sharpness differ by coaching
cohort, and how should the current-season games be weighted among themselves.

Proxy: for each (defense team, season, position, stat), the weekly
allowed-per-game series -> a rating vs the season's cross-defense mean.
Out of sample within season: blend [weeks 1..n, recency-weighted] with
[prior-year full season] and score |blend - actual weeks (n+1)..18|.

Axes:
  M1  prior_games  x  season-decay (n_full, n_zero, late_floor, curve)
  M2  recency form (geometric d / anchor+trend beta / window K,floor)
  M3  coaching cohort (none / dc_only / hc_only / both ; see data.coaching_changes)

    python scripts/sweep_defense_blend_design.py --years 2016-2025
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
COHORTS = ("all", "none", "dc_only", "hc_only", "both")
# stats that actually persist year-to-year (skip pure-noise TD rates)
STATS = {
    "QB": ["attempts", "completions", "passing_yards"],
    "RB": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards"],
    "TE": ["targets", "receptions", "receiving_yards"],
}
_ALLCOLS = sorted({c for v in STATS.values() for c in v})
NCUT = range(2, 15)                       # week cutoffs to blend from
PHASES = [("early n2-5", 2, 5), ("mid n6-9", 6, 9), ("late n10-14", 10, 14)]


# --- recency forms: weight for a game `g` weeks back (g>=1), given n games ---
def _recency_weights(games_ago, form):
    g = np.asarray(games_ago, dtype=float)
    kind = form[0]
    if kind == "geom":
        return form[1] ** (g - 1.0)
    if kind == "anchor":            # beta*flat + (1-beta)*geometric
        beta, d = form[1], form[2]
        flat = np.ones_like(g) / max(len(g), 1)
        geom = d ** (g - 1.0)
        geom = geom / geom.sum() if geom.sum() else geom
        return beta * flat + (1.0 - beta) * geom
    if kind == "window":           # last K games = 1, older = floor
        K, floor = form[1], form[2]
        return np.where(g <= K, 1.0, floor)
    raise ValueError(form)


# --- season-decay on the PRIOR weight: multiplier in [floor_frac, 1] ---
def _season_decay(n, cfg):
    if cfg is None:
        return 1.0
    n_full, n_zero, floor_frac, curve = cfg
    if n <= n_full:
        return 1.0
    if n >= n_zero:
        return floor_frac
    t = (n - n_full) / float(n_zero - n_full)      # 0..1
    if curve == "linear":
        f = t
    elif curve == "ease_in":
        f = t * t
    elif curve == "ease_out":
        f = 1.0 - (1.0 - t) ** 2
    elif curve == "step":
        f = 0.0 if t < 0.5 else 1.0
    else:
        f = t
    return 1.0 + f * (floor_frac - 1.0)


NAMED_BUILDS = {
    #                pg   season-decay (nf, nz, floor, curve)          recency form
    "B0 status-quo": (12, None,                                        ("geom", 0.85)),
    "B1 gentle":     (12, (5, 13, 0.12, "linear"),                     ("geom", 0.85)),
    "B2 hard-cut":   (12, (4, 8, 0.03, "step"),                        ("geom", 0.85)),
    "B3 stable-avg": (16, (7, 16, 0.20, "ease_out"),                   ("anchor", 0.6, 0.85)),
    "B4 hot-hand":   (8,  (4, 10, 0.05, "ease_in"),                    ("geom", 0.72)),
    "B5 anchor+trend": (12, (5, 13, 0.12, "linear"),                   ("anchor", 0.5, 0.82)),
    "B6 window":     (12, (5, 13, 0.12, "linear"),                     ("window", 4, 0.30)),
}
# Pass-A grid
PG_GRID = [8, 12, 16]
DECAY_GRID = [
    ("none", None),
    ("gentle", (5, 13, 0.15, "linear")),
    ("medium", (5, 11, 0.10, "ease_in")),
    ("sharp", (4, 9, 0.05, "step")),
    ("verysharp", (3, 7, 0.02, "step")),
]


def _load(years):
    ps = nfl.load_player_stats(seasons=list(years)).to_pandas()
    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]
    keep = ["player_id", "position", "season", "week", "team"] + _ALLCOLS
    ps = ps[[c for c in keep if c in ps.columns]].copy()
    for c in _ALLCOLS:
        if c not in ps.columns:
            ps[c] = 0.0
    for c in ps.columns:
        if c not in ("player_id", "position", "team"):
            ps[c] = pd.to_numeric(ps[c], errors="coerce")
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


def _weekly_allowed(df, pos, stat):
    """(def_team, season, week) -> allowed total of `stat` to `pos`."""
    d = df[df["position"] == pos]
    g = d.groupby(["def_team", "season", "week"])[stat].sum().reset_index()
    return g


def _rating_from_weeks(wk_df, wlo, whi, form):
    """recency-weighted mean allowed over [wlo,whi], / cross-defense mean."""
    d = wk_df[(wk_df["week"] >= wlo) & (wk_df["week"] <= whi)]
    if d.empty:
        return pd.Series(dtype=float)
    out = {}
    for tm, grp in d.groupby("def_team"):
        weeks = grp["week"].to_numpy(dtype=float)
        vals = grp[grp.columns[-1]].to_numpy(dtype=float)
        ga = (whi - weeks) + 1.0
        w = _recency_weights(ga, form)
        if w.sum() <= 0:
            out[tm] = np.nan
        else:
            out[tm] = float(np.average(vals, weights=w))
    s = pd.Series(out)
    lg = s.mean()
    return s / lg if lg else s * np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2016-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))

    df = _load(range(lo - 1, hi + 1))
    coh = {(int(r.season), r.team): r.coaching_cohort
           for r in coaching_change_table(lo - 1, hi).itertuples(index=False)}

    def _coh(season, team):
        c = coh.get((season, team), "unknown")
        return "none" if c == "dc_to_hc" else c

    # pre-slice weekly-allowed per (pos, stat, season)
    weekly = {}
    for pos in POSITIONS:
        for stat in STATS[pos]:
            g = _weekly_allowed(df, pos, stat)
            for y in range(lo - 1, hi + 1):
                weekly[(pos, stat, y)] = g[g["season"] == y].drop(columns="season")

    def _score(pg, decay_cfg, form, want_oos=False):
        """returns dict[(cohort, pos, stat, phase)] -> mean abs err ; and an
        optional odd/even split for the OOS check."""
        acc = {}
        acc_oe = {}
        for pos in POSITIONS:
            for stat in STATS[pos]:
                for y in years:
                    cur = weekly.get((pos, stat, y))
                    prv = weekly.get((pos, stat, y - 1))
                    if cur is None or prv is None or cur.empty or prv.empty:
                        continue
                    prior = _rating_from_weeks(prv, 1, 18, ("geom", 1.0))  # prior = flat full-season
                    for n in NCUT:
                        td = _rating_from_weeks(cur, 1, n, form)
                        fu = _rating_from_weeks(cur, n + 1, 18, ("geom", 1.0))
                        common = td.index.intersection(fu.index).intersection(prior.index)
                        a0 = n / (n + pg)
                        pw = (1.0 - a0) * _season_decay(n, decay_cfg)
                        phase = next((nm for nm, plo, phi in PHASES if plo <= n <= phi), None)
                        for tm in common:
                            c0, f0, p0 = td[tm], fu[tm], prior[tm]
                            if not (np.isfinite(c0) and np.isfinite(f0) and np.isfinite(p0)):
                                continue
                            blen = (1.0 - pw) * c0 + pw * p0
                            e = abs(blen - f0)
                            ch = _coh(y, tm)
                            for cc in ("all", ch):
                                acc.setdefault((cc, pos, stat, "ALL"), []).append(e)
                                if phase:
                                    acc.setdefault((cc, pos, stat, phase), []).append(e)
                            if want_oos:
                                key = (ch, "odd" if y % 2 else "even")
                                acc_oe.setdefault(key, []).append(e)
        mae = {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= 40}
        oe = {k: float(np.mean(v)) for k, v in acc_oe.items() if len(v) >= 40}
        return mae, oe

    base_mae, _ = _score(12, None, ("geom", 0.85))
    print(f"defense-blend design sweep {years[0]}-{years[-1]}\n"
          f"baseline = B0 (pg12, no decay, geom 0.85). numbers are MAE - baseMAE "
          f"(negative = better).\n", flush=True)

    # score each Pass-A config ONCE (cohort/pos/stat all come out of one scan)
    passA = {}
    for pg in PG_GRID:
        for lab, dc in DECAY_GRID:
            passA[(pg, lab)], _ = _score(pg, dc, ("geom", 0.85))
            print(f"  scored pg{pg}/{lab}", flush=True)

    print("\n" + "=" * 100)
    print("PASS A - prior_games x season-decay  (recency fixed at geom 0.85); "
          "cell = best-over-pg dMAE for that (cohort, decay)")
    print("=" * 100)
    for pos in POSITIONS:
        for stat in STATS[pos]:
            print(f"\n--- {pos} {stat} ---")
            print(f"{'cohort':<9}" + "".join(f"{lab[:9]:>10}" for lab, _ in DECAY_GRID))
            for cc in COHORTS:
                cells, best = [], (1e9, None)
                for lab, _dc in DECAY_GRID:
                    vals = []
                    for pg in PG_GRID:
                        m = passA[(pg, lab)]
                        key = (cc, pos, stat, "ALL")
                        if key in m and key in base_mae:
                            vals.append((m[key] - base_mae[key], pg))
                    if vals:
                        d, pg = min(vals)
                        cells.append(f"{d:+.4f}")
                        if d < best[0]:
                            best = (d, f"pg{pg}/{lab}")
                    else:
                        cells.append(f"{'-':>9}")
                print(f"{cc:<9}" + "".join(f"{c:>10}" for c in cells)
                      + (f"   best={best[1]} ({best[0]:+.4f})" if best[1] else ""))

    # ---- Pass C: named builds, per cohort/stat ----
    print("\n\n" + "=" * 100)
    print("PASS C - named builds vs B0, per cohort/stat  (dMAE)")
    print("=" * 100)
    build_scores = {}
    for name, cfg in NAMED_BUILDS.items():
        build_scores[name] = _score(*cfg)[0]
        print(f"  scored {name}", flush=True)
    for pos in POSITIONS:
        for stat in STATS[pos]:
            print(f"\n--- {pos} {stat} ---")
            print(f"{'build':<16}" + "".join(f"{cc:>10}" for cc in COHORTS))
            for name in NAMED_BUILDS:
                m = build_scores[name]
                row = []
                for cc in COHORTS:
                    key = (cc, pos, stat, "ALL")
                    if key in m and key in base_mae:
                        row.append(f"{m[key] - base_mae[key]:+.4f}")
                    else:
                        row.append(f"{'-':>9}")
                print(f"{name:<16}" + "".join(f"{c:>10}" for c in row))

    # ---- coaching OOS check: does `both` want a sharper decay than `none`? ----
    print("\n\n" + "=" * 100)
    print("COACHING OOS CHECK - best (pg,decay) per cohort fit on ODD seasons, scored on EVEN")
    print("=" * 100)
    oe_all = {}
    for pg in PG_GRID:
        for lab, dc in DECAY_GRID:
            _, oe = _score(pg, dc, ("geom", 0.85), want_oos=True)
            oe_all[(pg, lab)] = oe
    b0_oe = oe_all[(12, "none")]
    for cc in ("none", "dc_only", "hc_only", "both"):
        best = (1e9, None)
        for (pg, lab), oe in oe_all.items():
            v = oe.get((cc, "odd"))
            if v is not None and v < best[0]:
                best = (v, (pg, lab))
        if best[1] is None:
            print(f"  {cc:<9} (insufficient data)")
            continue
        pg, lab = best[1]
        be = oe_all[(pg, lab)].get((cc, "even"))
        b0e = b0_oe.get((cc, "even"))
        tag = f"  even-season dMAE vs B0 = {be - b0e:+.4f}" if (be and b0e) else "  (no even-season n)"
        print(f"  {cc:<9} fit-on-odd best = pg{pg}/{lab:<10}{tag}")


if __name__ == "__main__":
    main()
