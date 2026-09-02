"""Backfill NFL offensive & defensive coordinators for the years the committed
Ourlads archive doesn't cover (Ourlads is 2022-2025).

Source: the Wikipedia articles "List of current NFL {offensive,defensive}
coordinators" - a single, well-maintained wikitable per list. We pull the
article's REVISION as it stood in-season each year (via the MediaWiki API,
prop=revisions), so one snapshot per (list, year) gives all 32 teams at once
with a clean `| [[Team]] || '''[[Coach]]''' || {{nfly|year_hired}} || ...`
row format. ~20 API calls total for a decade, throttled with maxlag.

`year_hired` from the table is kept alongside our own name-diff so a human
can sanity-check "did this DC actually change" two independent ways.

    python -m data.coaching_history_wikipedia --years 2013-2021
writes external_data/coaching_coordinators_wikipedia_<lo>-<hi>.csv and prints
an eyeball table.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import requests

_UA = {"User-Agent": "NFLScholar-research/1.0 (personal non-commercial NFL projection model)"}
_API = "https://en.wikipedia.org/w/api.php"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_data")

_LISTS = {
    "dc": "List of current NFL defensive coordinators",
    "oc": "List of current NFL offensive coordinators",
}

# Wikipedia team-article name -> nflverse abbr (covers the relocation-era names
# these historical revisions use).
_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Los Angeles Rams": "LA", "St. Louis Rams": "LA",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

_TEAM_CELL = r"\|\s*\[\[([A-Za-z .0-9'’&-]+?)(?:\|[^\]]*)?\]\]\s*\|\|\s*"
# name cell can be: '''[[First Last]]''' | [[First Last|disp]] | {{sortname|First|Last}} | plain First Last
_NAME_CELL = (r"(?:'''\s*)?"
              r"(?:\[\[([A-Za-z .0-9'’.\-]+?)(?:\|[^\]]*)?\]\]"
              r"|\{\{sortname\|([A-Za-z.'’-]+)\|([A-Za-z.'’ -]+?)(?:\|[^}]*)?\}\}"
              r"|([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){1,3}))"
              r"\s*(?:''')?")
_YEAR = r"[^|]*?(?:\{\{nfly\|(\d{4})\}\}|\b(20\d{2})\b)?"
_ROW = re.compile(_TEAM_CELL + _NAME_CELL + r"\s*\|\|" + _YEAR)


def _rev_content_at(title: str, iso: str, session: requests.Session) -> tuple[str, str]:
    for attempt in range(4):
        r = session.get(_API, params={
            "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
            "rvprop": "content|timestamp", "rvslots": "main", "rvstart": iso,
            "rvdir": "older", "format": "json", "maxlag": 5}, headers=_UA, timeout=30)
        if r.status_code == 200 and "error" not in r.json():
            pg = next(iter(r.json()["query"]["pages"].values()))
            rev = pg.get("revisions", [{}])[0]
            return rev.get("timestamp", ""), rev.get("slots", {}).get("main", {}).get("*", "")
        time.sleep(2 + 3 * attempt)
    return "", ""


def _clean(name: str) -> str:
    name = re.sub(r"\s*\((American football|coach|interim)[^)]*\)", "", name, flags=re.I)
    return name.strip(" '’")


def parse_list(wikitext: str) -> dict[str, tuple[str, int | None]]:
    """team abbr -> (coach name, year_hired or None)."""
    out: dict[str, tuple[str, int | None]] = {}
    for m in _ROW.finditer(wikitext):
        team_name, link, sn_first, sn_last, plain, y1, y2 = m.groups()
        abbr = _NAME_TO_ABBR.get(team_name.strip())
        if not abbr:
            continue
        coach = link or (f"{sn_first} {sn_last}" if sn_first else None) or plain or ""
        coach = _clean(coach)
        if not coach or coach.upper() in {"TBA", "VACANT", "TBD", "N/A"}:
            coach = "Vacant"
        yr = int(y1 or y2) if (y1 or y2) else None
        # a table can list a team twice (interim mid-season); keep the first
        # (the season-opening staff, which is the pre-season prior-trust anchor)
        out.setdefault(abbr, (coach, yr))
    return out


def build(year_lo: int, year_hi: int) -> list[dict]:
    session = requests.Session()
    rows: list[dict] = []
    for role, title in _LISTS.items():
        for year in range(year_lo, year_hi + 1):
            # MID-season snapshot (early December): the table is fully
            # populated for season Y, but the January free-agent-coordinator
            # churn that would list season Y+1's incoming staff has not started
            # yet. A {year+1}-01 revision was contaminated with next year's
            # hires (BUF 2016 showed Frazier, not Thurman, etc.).
            ts, wt = _rev_content_at(title, f"{year}-12-05T00:00:00Z", session)
            parsed = parse_list(wt) if wt else {}
            print(f"  {role} {year}: rev {ts[:10] or 'FAIL'}, {len(parsed)} teams", file=sys.stderr)
            for abbr, (coach, yr_hired) in parsed.items():
                rows.append({"season": year, "team": abbr, "role": role,
                             "coach": coach, "year_hired": yr_hired, "rev": ts[:10]})
            time.sleep(1.0)
    return rows


def out_path(year_lo: int, year_hi: int) -> str:
    return os.path.join(OUT_DIR, f"coaching_coordinators_wikipedia_{year_lo}-{year_hi}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2013-2021")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    rows = build(lo, hi)
    path = out_path(lo, hi)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["season", "team", "role", "coach", "year_hired", "rev"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["role"], r["season"], r["team"])))
    print(f"\nwrote {len(rows)} rows -> {path}\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    dc = df[df.role == "dc"].pivot_table(index="team", columns="season", values="coach", aggfunc="first")
    print("=== DEFENSIVE COORDINATOR by team-season (eyeball me) ===")
    with pd.option_context("display.max_rows", 40, "display.max_columns", 20, "display.width", 260):
        print(dc.to_string())
    per_year = df[df.role == "dc"].groupby("season")["team"].nunique()
    print("\nDC coverage by season:", per_year.to_dict())


if __name__ == "__main__":
    main()
