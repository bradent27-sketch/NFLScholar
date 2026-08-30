"""
Import fresh Ourlads depth-chart pages from a folder (default:
external_data/ourlads_inbox/) into the live snapshot the weekly model reads.

Same code path as the Depth Charts tab's importer, plus:
  - the previous snapshot csv AND the raw pages consumed are copied into a
    timestamped folder under external_data/ourlads_archive/ (so a past-week
    analysis can pin the exact chart the model saw);
  - after a clean import the processed inbox files are MOVED into that same
    archive folder, leaving the inbox empty and ready for the next update.

NOTE ON CACHES: this clears the build cache in THIS process only. A separately
running Streamlit app will not see the new chart until you re-import through
its "Import from ourlads_inbox/" button, hit "Clear cache", or restart it.

Usage:
    python scripts/import_ourlads.py                 # inbox, current season
    python scripts/import_ourlads.py --year 2026
    python scripts/import_ourlads.py --dir "some/other/folder" --keep   # don't move files
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ourlads_depth_charts import (  # noqa: E402
    OURLADS_INBOX_DIR, _current_nfl_season, save_ourlads_snapshot_from_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OURLADS_INBOX_DIR), help="folder of saved Ourlads pages")
    ap.add_argument("--year", type=int, default=None, help="season (default: current)")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT move processed files out of the folder afterwards")
    args = ap.parse_args()

    year = args.year or _current_nfl_season()
    src = Path(args.dir)
    print(f"importing Ourlads pages from {src}  (season {year})")

    snapshot, report = save_ourlads_snapshot_from_dir(src, year=year)
    if report.get("error"):
        print(f"  ERROR: {report['error']}")
        return 1

    print(f"  parsed {len(snapshot):,} rows for {report['team_count']}/32 teams "
          f"from {len(report.get('source_files', []))} files")
    if report.get("missing_teams"):
        print(f"  still missing: {', '.join(report['missing_teams'])}")
    for bad in report.get("unreadable_files", []):
        print(f"  skipped {bad.get('source_file')}: {bad.get('error')}")
    arch = report.get("archive") or {}
    if arch.get("archived_csv"):
        print(f"  archived previous snapshot -> {arch['archived_csv']}")
    if arch.get("archived_pages"):
        print(f"  archived {arch['archived_pages']} raw page(s) under {arch['archive_dir']}")
    print(f"  wrote {report.get('path')}")

    # Move the processed inbox files into the archive batch so the inbox is
    # clean for next time (only when importing FROM the inbox, not --dir).
    if not args.keep and src.resolve() == Path(OURLADS_INBOX_DIR).resolve():
        dest = Path(arch.get("archive_dir") or (src.parent / "ourlads_archive")) / "consumed_inbox"
        moved = 0
        for name in report.get("source_files", []):
            f = src / name
            if f.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest / name))
                moved += 1
        if moved:
            print(f"  moved {moved} processed file(s) out of the inbox -> {dest}")

    try:
        from data.weekly_projections import build_weekly_projections
        build_weekly_projections.clear()
        print("  cleared this process's build cache")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
