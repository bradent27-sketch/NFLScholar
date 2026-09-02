"""Season-over-season NFL coaching-change labels.

Two verified sources, deliberately no hand-typed history (a wrong coaching
label silently poisons any prior-season-trust study that keys off it):

  * HEAD COACH  - nflverse `load_schedules()` `home_coach` / `away_coach`.
    Zero nulls 1999-2026, so `hc_changed` is available for every transition.
  * OFF / DEF COORDINATOR - the committed Ourlads pre-Week-1 archive
    (external_data/ourlads_coaching_staff_by_season.csv), 2022-2025 only.
    `oc_changed` / `dc_changed` therefore exist only for 2023, 2024, 2025.

A "change" compares season Y's staff to season Y-1's for the SAME team. The
first season we have data for a team has NaN change flags (no baseline). A
"Vacant" coordinator (Ourlads' literal value - e.g. BUF 2023, run by the head
coach) is normalized to the string "Vacant" and counts as different from a
named predecessor.

`coaching_change_table(...)` returns one row per (season, team):
    season, team, hc, oc, dc,
    hc_changed, oc_changed, dc_changed,      # nullable bool
    def_staff_changed,                        # hc_changed | dc_changed
    coaching_cohort                           # 'none' | 'dc_only' | 'hc_only' | 'both' | 'unknown'
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd

_EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_data")
_OURLADS_STAFF = os.path.join(_EXT, "ourlads_coaching_staff_by_season.csv")
# Wikipedia coordinator-list backfill for the pre-Ourlads years (written by
# data/coaching_history_wikipedia.py). ~75-85% team-season coverage; the gaps
# just fall through to the 'unknown' cohort, never a wrong label.
_WIKI_COORD_GLOB = "coaching_coordinators_wikipedia_"

# nflverse schedules use era-accurate abbreviations; fold the relocations onto
# the modern key so a franchise's history is one series.
_ABBR_CANON = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}


def _canon_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().replace(_ABBR_CANON)


def _norm_name(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
            .str.replace(r"\.", "", regex=True)
            .str.replace(r"\b(jr|sr|ii|iii|iv)\b", "", regex=True, case=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold())


@functools.lru_cache(maxsize=8)
def _head_coach_by_season_team(season_lo: int, season_hi: int) -> pd.DataFrame:
    """Long (season, team, hc) from nflverse schedules - regular season only,
    one coach per team-season (the modal coach if a team changed mid-year,
    which is the right anchor for a *pre-season* prior-trust decision)."""
    import nflreadpy as nfl
    seasons = list(range(season_lo, season_hi + 1))
    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched = sched[sched["game_type"].astype(str).str.upper().isin({"REG", "REGULAR", ""})
                  | sched["game_type"].isna()]
    home = sched[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "hc"})
    away = sched[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "hc"})
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["team", "hc"])
    long["team"] = long["team"].astype(str)
    modal = (long.groupby(["season", "team"])["hc"]
             .agg(lambda s: s.value_counts().idxmax()).reset_index())
    return modal


@functools.lru_cache(maxsize=2)
def _ourlads_staff() -> pd.DataFrame:
    if not os.path.exists(_OURLADS_STAFF):
        return pd.DataFrame(columns=["season", "team", "oc", "dc"])
    df = pd.read_csv(_OURLADS_STAFF).rename(columns={"year": "season"})
    df["team"] = _canon_team(df["team"])
    df["dc_source"] = "ourlads"
    return df[["season", "team", "oc", "dc", "dc_source"]]


@functools.lru_cache(maxsize=2)
def _wiki_coordinators() -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(os.path.join(_EXT, _WIKI_COORD_GLOB + "*.csv")))
    if not files:
        return pd.DataFrame(columns=["season", "team", "oc", "dc", "dc_source"])
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["team"] = _canon_team(raw["team"])
    dc = raw[raw["role"] == "dc"][["season", "team", "coach"]].rename(columns={"coach": "dc"})
    oc = raw[raw["role"] == "oc"][["season", "team", "coach"]].rename(columns={"coach": "oc"})
    out = dc.merge(oc, on=["season", "team"], how="outer")
    # Drop a "DC == that year's HC" parse artifact UNLESS the same name is the
    # dc==hc for 2+ consecutive years for that team (a real HC-run defense,
    # e.g. NE / Belichick 2019-2021 - informative "no change").
    hc = _head_coach_by_season_team(int(out["season"].min()) - 1, int(out["season"].max()))
    hc["team"] = _canon_team(hc["team"])
    out = out.merge(hc, on=["season", "team"], how="left")
    same = out["dc"].notna() & (_norm_name(out["dc"]) == _norm_name(out["hc"].fillna("")))
    out = out.sort_values(["team", "season"])
    run = same & (same.groupby(out["team"]).transform(
        lambda s: s & (s.shift(1, fill_value=False) | s.shift(-1, fill_value=False))))
    out.loc[same & ~run, "dc"] = np.nan
    out["dc_source"] = np.where(out["dc"].notna(), "wikipedia", None)
    return out[["season", "team", "oc", "dc", "dc_source"]]


# Hand-verified single-cell fills for the specific (season, team) chain links
# the Wikipedia coordinator-list scrape missed (2019-2022), transcribed from
# pro-football-history.com per-team DC-history pages (franchpos/<id>/8/<slug>).
# Same long schema as the Wikipedia CSVs. Only the exact gaps the 2020-2025
# cohort study needed - not a general backfill.
_MANUAL_COORD_GLOB = "coaching_coordinators_manual_"

# dc_source -> merge priority (lower wins). Ourlads is the verified pre-Week-1
# archive; the manual pro-football-history fills are hand-checked against the
# franchise pages; Wikipedia is the bulk scrape with the widest coverage but
# the most parse noise.
_DC_SOURCE_RANK = {"ourlads": 0, "pro-football-history": 1, "wikipedia": 2}


@functools.lru_cache(maxsize=2)
def _manual_coordinators() -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(os.path.join(_EXT, _MANUAL_COORD_GLOB + "*.csv")))
    if not files:
        return pd.DataFrame(columns=["season", "team", "oc", "dc", "dc_source"])
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["team"] = _canon_team(raw["team"])
    dc = raw[raw["role"] == "dc"][["season", "team", "coach"]].rename(columns={"coach": "dc"})
    oc = raw[raw["role"] == "oc"][["season", "team", "coach"]].rename(columns={"coach": "oc"})
    out = dc.merge(oc, on=["season", "team"], how="outer")
    out["dc_source"] = np.where(out["dc"].notna(), "pro-football-history", None)
    if "oc" not in out.columns:
        out["oc"] = np.nan
    return out[["season", "team", "oc", "dc", "dc_source"]]


@functools.lru_cache(maxsize=2)
def _coordinator_history() -> pd.DataFrame:
    """One row per (season, team) -> oc, dc, dc_source. Priority when more than
    one source has a cell: Ourlads (2022+, verified) > pro-football-history
    manual gap fills > Wikipedia bulk scrape (see _DC_SOURCE_RANK)."""
    parts = [_ourlads_staff(), _manual_coordinators(), _wiki_coordinators()]
    allrows = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    allrows = allrows[allrows["dc"].notna()].copy()
    allrows["_rank"] = allrows["dc_source"].map(_DC_SOURCE_RANK).fillna(9)
    both = (allrows.sort_values(["season", "team", "_rank"])
            .drop_duplicates(["season", "team"], keep="first")
            .drop(columns="_rank")
            .sort_values(["season", "team"]).reset_index(drop=True))
    return both


def _changed(cur: pd.Series, prev: pd.Series) -> pd.Series:
    both_known = cur.notna() & prev.notna()
    return pd.Series(np.where(both_known, _norm_name(cur).values != _norm_name(prev).values, np.nan),
                     index=cur.index, dtype="object")


@functools.lru_cache(maxsize=16)
def coaching_change_table(season_lo: int = 2016, season_hi: int = 2026) -> pd.DataFrame:
    """One row per (season, team): hc/oc/dc names, change flags, and a
    `coaching_cohort` for the defense-prior study. HC is nflverse (all years);
    DC is Ourlads 2022+ / Wikipedia backfill earlier, so `dc_changed` is only
    non-null where BOTH that season and the prior one have a known DC.

    Cohorts:
      none      - no HC and no DC change
      dc_only   - DC changed, HC stayed
      hc_only   - HC changed, DC stayed (rare)
      both      - HC and DC both changed
      dc_to_hc  - the prior year's DC IS this year's HC (same team): the
                  defensive brain was promoted, so scheme continuity holds
                  even though both title slots technically turned over
      unknown   - missing DC data for this season or the prior one
    """
    hc = _head_coach_by_season_team(season_lo - 1, season_hi)
    hc["team"] = _canon_team(hc["team"])
    hc = hc.sort_values(["team", "season"])
    hc["hc_prev"] = hc.groupby("team")["hc"].shift(1)
    hc["hc_changed"] = _changed(hc["hc"], hc["hc_prev"])

    coord = _coordinator_history().sort_values(["team", "season"]).copy()
    coord["dc_prev"] = coord.groupby("team")["dc"].shift(1)
    coord["oc_prev"] = coord.groupby("team")["oc"].shift(1)
    coord["dc_changed"] = _changed(coord["dc"], coord["dc_prev"])
    coord["oc_changed"] = _changed(coord["oc"], coord["oc_prev"])

    out = hc[["season", "team", "hc", "hc_prev", "hc_changed"]].copy()
    out = out[(out["season"] >= season_lo) & (out["season"] <= season_hi)]
    out = out.merge(coord[["season", "team", "oc", "dc", "dc_prev", "dc_source",
                           "oc_changed", "dc_changed"]],
                    on=["season", "team"], how="left")

    # dc_to_hc: prior-season DC (any source) == this-season HC, same team.
    out["dc_to_hc"] = (out["dc_prev"].notna() & out["hc"].notna()
                       & (_norm_name(out["dc_prev"].fillna("")) == _norm_name(out["hc"].fillna(""))))

    def _cohort(row):
        if row["dc_to_hc"]:
            return "dc_to_hc"
        hc_c, dc_c = row["hc_changed"], row["dc_changed"]
        if pd.isna(dc_c) or pd.isna(hc_c):
            return "unknown"
        hc_c, dc_c = bool(hc_c), bool(dc_c)
        if hc_c and dc_c:
            return "both"
        if dc_c:
            return "dc_only"
        if hc_c:
            return "hc_only"
        return "none"

    out["def_staff_changed"] = np.where(
        out["hc_changed"].isna() & out["dc_changed"].isna(), np.nan,
        (out["hc_changed"].fillna(False).astype(bool)
         | out["dc_changed"].fillna(False).astype(bool)))
    out["coaching_cohort"] = out.apply(_cohort, axis=1)
    return out.sort_values(["season", "team"]).reset_index(drop=True)


def eyeball_table(season_lo: int = 2016, season_hi: int = 2025) -> "pd.DataFrame":
    """Wide HC / DC / cohort view for a human sanity check."""
    t = coaching_change_table(season_lo, season_hi)
    t["cell"] = t["hc"].fillna("?") + "  |  " + t["dc"].fillna("?") + "  [" + t["coaching_cohort"] + "]"
    return t.pivot_table(index="team", columns="season", values="cell", aggfunc="first")


# Per-cohort DEFENSE_PRIOR_GAMES for v2_coaching_aware_defense_prior. Derived
# from scripts/analyze_coaching_defense_prior.py on 2018-2025 (HC) / 2023-2025
# (DC): year-over-year defense-allowed persistence AND the out-of-sample
# optimal blend weight both say -
#   * a wholesale defensive-staff change (HC + DC) resets the unit -> trust
#     the prior season LESS (persistence corr +0.02; optimal prior_games ~8).
#   * a coordinator-only change with the HC retained is usually an internal
#     promotion / same system -> the prior season is MORE predictive than
#     average (persistence corr +0.33, the highest of any cohort; optimal
#     prior_games 20-40, stable in direction every season). Implemented
#     conservatively at 18, well short of what the raw curve wants.
#   * no change / unknown -> the shipped default (DEFENSE_PRIOR_GAMES).
# `hc_only` (new HC, same DC) is too rare to fit (4 team-seasons); a new HC
# almost always reshapes the unit, so it is treated like `both`.
_COHORT_DEFAULTS = {
    "none": None,        # -> caller's DEFENSE_PRIOR_GAMES default
    "unknown": None,
    "dc_to_hc": None,   # DC promoted to HC same team = scheme continuity -> treat as 'none'
    "dc_only": 18.0,
    "hc_only": 8.0,
    "both": 8.0,
}


def _cohort_prior_games() -> dict:
    """Per-cohort DEFENSE_PRIOR_GAMES, env-overridable for the sweep
    (COHORT_PRIOR_GAMES="dc_only=20,both=8,dc_to_hc=6,hc_only=8"). An unset or
    "none"/"default" value for a cohort keeps the caller's scalar default."""
    out = dict(_COHORT_DEFAULTS)
    spec = os.environ.get("COHORT_PRIOR_GAMES", "")
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in out:
            continue
        v = v.strip().lower()
        out[k] = None if v in {"", "none", "default"} else float(v)
    return out


COHORT_DEFENSE_PRIOR_GAMES = _cohort_prior_games()


# Per-POSITION x cohort DEFENSE_PRIOR_GAMES. ALL None = every cohort takes the
# caller's scalar default, i.e. this table is a no-op.
#
# REVERTED TO NEUTRAL 2026-09-01, AND DELIBERATELY SO. A fitted table did live
# here (QB none=20/both=8, WR none=26/both=8, TE dc_only=18/both=16, ...),
# derived from scripts/analyze_coaching_defense_prior_bystat.py. It was
# DISPROVEN and must not be restored from git history without re-testing:
#
#   * The weeks-2-9 model A/B that motivated it showed START-QB -0.016 and
#     START-TE -0.039 (CI-excl-0) - but that was 2 significant cells out of a
#     63-cell grid, which is the false-positive count you expect at 5%.
#   * The confirm run refused to replicate it. Weeks 10-17 flipped sign
#     (START-WR +0.020, CI-excl-0, i.e. actively WORSE); weeks 4-13 came back
#     flat with every CI spanning 0.
#   * The per-stat 10-season design sweep (scripts/sweep_defense_blend_design.py)
#     found the cohort preference in-sample, then lost it out-of-sample: fit on
#     odd seasons and scored on even, EVERY cohort's best config was "no decay"
#     with an improvement of +0.000/+0.0001/+0.0019/+0.0000.
#   * Cohort sample sizes make fitting 16 cells hopeless anyway: hc_only is
#     ~44 team-seasons across a decade, both ~188.
#
# The cohort machinery below is retained because the DESCRIPTIVE finding is
# real (a HC change cuts year-over-year defense persistence from ~+0.23 to
# ~+0.14 correlation) and because it is the starting point if this is ever
# revisited via defensive PERSONNEL continuity rather than coach names. Every
# cell stays env-overridable (POS_COHORT_PRIOR_GAMES) so a sweep can still
# explore it without shipping anything.
_POS_COHORT_DEFAULTS = {
    "QB": {"none": None, "dc_only": None, "both": None, "hc_only": None, "dc_to_hc": None, "unknown": None},
    "RB": {"none": None, "dc_only": None, "both": None, "hc_only": None, "dc_to_hc": None, "unknown": None},
    "WR": {"none": None, "dc_only": None, "both": None, "hc_only": None, "dc_to_hc": None, "unknown": None},
    "TE": {"none": None, "dc_only": None, "both": None, "hc_only": None, "dc_to_hc": None, "unknown": None},
}


def _pos_cohort_prior_games() -> dict:
    """{POS: {cohort: prior_games or None}}, env-overridable per cell with
    POS_COHORT_PRIOR_GAMES="QB:none=20,WR:both=8,RB:none=16" (comma-separated
    POS:cohort=value). "none"/"default"/"" for a cell -> caller's scalar
    default for that cell."""
    out = {p: dict(d) for p, d in _POS_COHORT_DEFAULTS.items()}
    spec = os.environ.get("POS_COHORT_PRIOR_GAMES", "")
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part or "=" not in part:
            continue
        key, v = part.split("=", 1)
        pos, coh = (key.split(":", 1) + [""])[:2]
        pos, coh = pos.strip().upper(), coh.strip().lower()
        if pos not in out or coh not in out[pos]:
            continue
        v = v.strip().lower()
        out[pos][coh] = None if v in {"", "none", "default"} else float(v)
    return out


def defense_prior_games_by_team(season: int, default_prior_games: float,
                                season_lo: int = 2016, pos: str = None) -> "pd.Series":
    """team -> prior_games to use for that team's DEFENSE this season, given
    its coaching-change cohort (and position, if given). Teams with no cohort
    signal (pre-2023, missing Ourlads row, or a season with no coordinator
    archive) get the default, so this is safe to apply unconditionally when the
    flag is on. With ``pos`` in {QB,RB,WR,TE} the per-position cohort table is
    used; otherwise the whole-pool cohort table."""
    tbl = coaching_change_table(season_lo, season)
    tbl = tbl[tbl["season"] == season]
    if pos is not None and str(pos).upper() in _POS_COHORT_DEFAULTS:
        pg_map = _pos_cohort_prior_games()[str(pos).upper()]
    else:
        pg_map = _cohort_prior_games()
    mapping = tbl.set_index("team")["coaching_cohort"].map(lambda c: pg_map.get(c))
    return mapping.where(mapping.notna(), float(default_prior_games)).astype(float)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)
    t = coaching_change_table(2017, 2026)
    print(t.to_string())
    print()
    print("HC-change rate by season:")
    print(t.groupby("season")["hc_changed"].mean(numeric_only=False).apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-"))
    print()
    print("cohort counts (2023-2025, where DC data exists):")
    print(t[t["coaching_cohort"] != "unknown"]["coaching_cohort"].value_counts())
