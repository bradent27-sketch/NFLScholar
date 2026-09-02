"""Importer for hand-saved Ourlads "Depth Chart Archive" HTML pages.

Input:  external_data/OurLads Historical Depth Charts - Week Before Season 0901/
        (recursively) - one Ourlads archive HTML page per team per season, saved
        at the ~09/01 pre-Week-1 snapshot. Seeded with 32 teams x 2022-2025;
        drop each new season's 32 files into the same folder (a per-year
        subfolder is fine) and re-run - the script keys everything off the
        archive date inside each file, not the count or the filename.

Each page carries a <tbody id="ctl00_phContent_dcTBody"> depth table plus a
<div class="dc-coaches"> HC/OC/DC/ST block. Parsed into two flat CSVs the app
can read directly:

  external_data/ourlads_depth_charts_history.csv
      year, archive_date, last_updated, team, team_ourlads, team_name,
      unit, slot, slot_rank, jersey, player_raw, player_last, player_first,
      acq_tag, caps_name
      unit in {OFF, DEF, ST, PS, RES}

  external_data/ourlads_coaching_staff_by_season.csv
      year, archive_date, last_updated, team, team_name, hc, oc, dc, st

The live in-season pipeline (data/ourlads_depth_charts.py, printer-friendly
.mhtml -> external_data/ourlads_depth_charts.csv) is a separate, volatile
current-week snapshot and is left alone; this file is the frozen archive.

Run:  python scripts/import_ourlads_historical.py
Prints a validation report (filename<->content year, pre-Week-1 date sanity,
duplicates, per-year coverage) and writes nothing if a hard inconsistency is
found unless --force is passed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(
    ROOT, "external_data", "OurLads Historical Depth Charts - Week Before Season 0901"
)
OUT_CHART = os.path.join(ROOT, "external_data", "ourlads_depth_charts_history.csv")
OUT_STAFF = os.path.join(ROOT, "external_data", "ourlads_coaching_staff_by_season.csv")

# Ourlads team-name -> nflverse abbreviation. Ourlads' own class codes (ARZ, RAM)
# are captured separately as team_ourlads.
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

SECTION_MAP = {
    "Offense": "OFF", "Defense": "DEF", "Special Teams": "ST",
    "Practice Squad": "PS", "Reserves": "RES", "Injured Reserve": "RES",
}

# Acquisition-tag shapes Ourlads uses in the 3rd column, e.g.:
#   22/1  (drafted 2022 rd 1)   U/Ten (UFA from TEN)   T/LAR (trade)
#   SF23  (street/veteran FA)   CF24  (college FA)     P/Phi (practice-squad add)
#   W/Cin (waiver)   RFA/D2 ...  D1b ...   plus IR / NFI / PUP status words
_TAG_RE = re.compile(
    r"^(?:\d{2}/\d{1,2}[a-z]?|[A-Z]{1,3}/[A-Za-z]{2,3}|(?:SF|CF|FA|PS|TR)\d{2}"
    r"|D\d[a-z]?|RFA.*|ERFA.*|UFA.*|IR.*|NFI.*|PUP.*|SUS.*|EXE.*|RET.*)$"
)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(fragment))).strip()


def _split_player(cell: str, unit: str):
    """'RAYMOND, KALIF U/Ten' -> ('Raymond', 'Kalif', 'U/Ten', caps=True).

    In PS/RES sections the trailing token is a POSITION (WR/QB/DE), not an
    acquisition tag - returned as acq_tag with a leading '=' so it is
    distinguishable downstream.
    """
    cell = _text(cell)
    if not cell:
        return ("", "", "", "", False)
    if "," in cell:
        last, rest = cell.split(",", 1)
    else:
        last, rest = cell, ""
    last = last.strip()
    toks = rest.strip().split()
    acq = ""
    if toks:
        tail = toks[-1]
        if unit in ("PS", "RES") and re.fullmatch(r"[A-Z]{1,3}", tail):
            acq = "=" + tail
            toks = toks[:-1]
        elif _TAG_RE.match(tail):
            acq = tail
            toks = toks[:-1]
    first = " ".join(toks).strip()
    # Ourlads renders some last names in ALL-CAPS. Empirically this tracks an
    # offseason roster/contract change (new acquisition, trade, notable
    # re-signing) rather than "projected starter" - it is NOT on rank 1 only.
    # Kept as a raw descriptive flag (`caps_name`); treat as a soft hint, not
    # a definition.
    caps = bool(last) and last == last.upper() and any(c.isalpha() for c in last)
    # normalise ALL-CAPS to title case for storage; keep the flag
    last_norm = last.title() if caps else last
    first_norm = first.title() if first == first.upper() and first else first
    return (last_norm, first_norm, acq, cell, caps)


def parse_file(path: str) -> dict:
    raw = open(path, encoding="utf-8", errors="replace").read()
    out: dict = {"path": path, "file": os.path.basename(path), "warnings": []}

    u = re.search(r'id="ctl00_phContent_DateUpd"[^>]*>\s*Last Updated:\s*(.+?)\s*</div>', raw)
    out["last_updated"] = _text(u.group(1)) if u else ""

    # Preferred: an archive page - "<h1 ...>Detroit Lions Archive (09/02/2024)</h1>".
    m = re.search(
        r'<h1 id="ctl00_phContent_TeamTitle"[^>]*class="([^"]*)"[^>]*>\s*'
        r"(.+?)\s+Archive\s*\((\d{2}/\d{2}/\d{4})\)\s*</h1>",
        raw,
    )
    if m:
        out["team_ourlads"] = m.group(1).strip()
        out["team_name"] = m.group(2).strip()
        out["archive_date"] = m.group(3).strip()
    else:
        # Fallback: a plain current depth-chart page - "<h1 ...>Detroit Lions</h1>",
        # no archive date. Take the team from the h1 and the snapshot date from
        # "Last Updated". Filename year (if any) backfills the year.
        m2 = re.search(
            r'<h1 id="ctl00_phContent_TeamTitle"[^>]*class="([^"]*)"[^>]*>\s*(.+?)\s*</h1>',
            raw,
        )
        if not m2:
            out["warnings"].append("no TeamTitle <h1> found - UNPARSED")
            return out
        out["team_ourlads"] = m2.group(1).strip()
        out["team_name"] = re.sub(r"\s+Archive.*$", "", m2.group(2).strip())
        d = re.search(r"(\d{2}/\d{2}/\d{4})", out["last_updated"])
        out["archive_date"] = d.group(1) if d else ""
        if not out["archive_date"]:
            out["warnings"].append("plain page with no parseable date - year will rely on filename")

    if out["archive_date"]:
        out["archive_year"] = int(out["archive_date"].split("/")[-1])
    out["team"] = NAME_TO_ABBR.get(out["team_name"], "")
    if not out["team"]:
        out["warnings"].append(f"unknown team name '{out['team_name']}'")

    for key, cid in (("hc", "liHC"), ("oc", "liOC"), ("dc", "liDC"), ("st", "liST")):
        c = re.search(rf'id="ctl00_phContent_{cid}"[^>]*>\s*[A-Z]+:\s*(.+?)\s*</li>', raw)
        out[key] = _text(c.group(1)) if c else ""

    tb = re.search(r'<tbody id="ctl00_phContent_dcTBody">(.*?)</tbody>', raw, re.S)
    rows = []
    if not tb:
        out["warnings"].append("no depth-chart <tbody> found")
        out["rows"] = rows
        return out

    section = None
    for attrs, body in re.findall(r"<tr([^>]*)>(.*?)</tr>", tb.group(1), re.S):
        if "row-dc-pos-mobile" in attrs:
            continue
        if "row-dc-pos" in attrs:
            label = _text(body)
            section = SECTION_MAP.get(label, label.upper()[:4] or None)
            continue
        if section is None:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        if len(cells) < 3:
            continue
        slot = _text(cells[0])
        if not slot:
            continue
        # cells: [slot] then (jersey, player) pairs
        pairs = cells[1:]
        rank = 0
        for i in range(0, len(pairs) - 1, 2):
            jersey = _text(pairs[i])
            player_cell = pairs[i + 1]
            if not _text(player_cell):
                continue
            rank += 1
            last, first, acq, rawname, caps = _split_player(player_cell, section)
            rows.append({
                "unit": section, "slot": slot, "slot_rank": rank,
                "jersey": jersey, "player_raw": rawname,
                "player_last": last, "player_first": first,
                "acq_tag": acq, "caps_name": int(caps),
            })
    out["rows"] = rows
    if not rows:
        out["warnings"].append("parsed 0 depth-chart rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--force", action="store_true",
                    help="write CSVs even if a hard inconsistency is found")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.htm*"), recursive=True))
    print(f"found {len(files)} files under {args.src}\n")
    if not files:
        return 2

    parsed, hard = [], []
    seen_team_year = defaultdict(list)

    for path in files:
        base = os.path.basename(path)
        # filename year: trailing "...Ourlads2024.html" OR a leading "2026 Team..."
        fn_year_m = re.search(r"Ourlad?s?\s*(\d{4})", base) or re.match(r"\s*(20\d{2})\b", base)
        fn_year = int(fn_year_m.group(1)) if fn_year_m else None
        rec = parse_file(path)
        rec["file_year"] = fn_year
        rec["year"] = rec.get("archive_year") or fn_year

        if rec["year"] is None or "team_name" not in rec:
            rec["warnings"].append("could not establish (team, year) - UNPARSED")
            hard.append(rec)
            parsed.append(rec)
            continue

        adate = rec.get("archive_date") or ""
        # filename year vs content archive-year
        if fn_year is not None and rec.get("archive_year") and fn_year != rec["archive_year"]:
            rec["warnings"].append(
                f"FILENAME year {fn_year} != archive-date year {rec['archive_year']}"
            )
            hard.append(rec)
        # pre-Week-1 sanity: expect ~09/01-09/07
        if adate:
            mm, dd, _ = adate.split("/")
            if mm != "09" or not (1 <= int(dd) <= 7):
                rec["warnings"].append(
                    f"archive date {adate} is not an early-September (pre-Week-1) snapshot"
                )
        else:
            rec["warnings"].append("no archive date in file - snapshot date left blank")
        seen_team_year[(rec.get("team", "?"), rec["year"])].append(base)
        parsed.append(rec)

    # ---- report ---------------------------------------------------------
    print("=" * 72)
    print("PER-FILE")
    print("=" * 72)
    for rec in parsed:
        tag = "  ok " if not rec["warnings"] else "WARN "
        team = rec.get("team", "??")
        yr = rec.get("year") or "????"
        ad = rec.get("archive_date") or "(no date)"
        nrows = len(rec.get("rows", []))
        print(f"{tag}{team:<4} {yr}  {ad:<12}  rows={nrows:<3}  {rec['file']}")
        for w in rec["warnings"]:
            print(f"       - {w}")

    print("\n" + "=" * 72)
    print("COVERAGE  (per season; a full league = 32 teams)")
    print("=" * 72)
    by_year = defaultdict(set)
    for (team, yr), fl in seen_team_year.items():
        by_year[yr].add(team)
        if len(fl) > 1:
            print(f"  DUPLICATE {team} {yr}: {fl}")
    for yr in sorted(by_year):
        missing = sorted(set(NAME_TO_ABBR.values()) - by_year[yr])
        print(f"  {yr}: {len(by_year[yr])} teams" + (f"  MISSING {missing}" if missing else "  (complete)"))

    staff_missing = [
        f"{r['team']} {r['year']}"
        for r in parsed
        if r.get("year") and not all([r.get("hc"), r.get("oc"), r.get("dc")])
    ]
    if staff_missing:
        print(f"\n  coaching staff incomplete for: {staff_missing}")

    total_rows = sum(len(r.get("rows", [])) for r in parsed)
    print(f"\n  total depth-chart rows parsed: {total_rows}")
    print(f"  hard inconsistencies: {len(hard)}")

    if hard and not args.force:
        print("\nNOT writing CSVs - resolve the hard inconsistencies above or pass --force.")
        return 1

    # ---- write --------------------------------------------------------
    with open(OUT_CHART, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "year", "archive_date", "last_updated", "team", "team_ourlads", "team_name",
            "unit", "slot", "slot_rank", "jersey", "player_raw",
            "player_last", "player_first", "acq_tag", "caps_name",
        ])
        for r in parsed:
            if not r.get("year") or not r.get("rows"):
                continue
            for row in r["rows"]:
                w.writerow([
                    r["year"], r.get("archive_date", ""), r["last_updated"],
                    r["team"], r["team_ourlads"], r["team_name"],
                    row["unit"], row["slot"], row["slot_rank"], row["jersey"],
                    row["player_raw"], row["player_last"], row["player_first"],
                    row["acq_tag"], row["caps_name"],
                ])

    with open(OUT_STAFF, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "archive_date", "last_updated", "team", "team_name",
                    "hc", "oc", "dc", "st"])
        for r in parsed:
            if not r.get("year") or "team_name" not in r:
                continue
            w.writerow([r["year"], r.get("archive_date", ""), r["last_updated"],
                        r["team"], r["team_name"],
                        r.get("hc", ""), r.get("oc", ""), r.get("dc", ""), r.get("st", "")])

    print(f"\nwrote {OUT_CHART}")
    print(f"wrote {OUT_STAFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
