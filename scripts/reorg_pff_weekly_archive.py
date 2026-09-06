"""Reorganise a raw PFF weekly export dump into the archive layout the
loader reads (2026-09-04).

The 2024/2025 weekly archives live at
    pff_imports/{year}/weekly/{week}/receiving_{summary,concept,scheme}.csv
plus a weekly/manifest.csv. The 2022/2023 exports the user just added are a
flat folder of numbered files instead:
    pff_imports/{year}/Weekly Data/receiving_{summary,concept,scheme}.csv        # week 1
    pff_imports/{year}/Weekly Data/receiving_{summary,concept,scheme} (N).csv    # week N+1

This copies them into the {week}/ layout, REGULAR SEASON ONLY (weeks 1-18),
and writes a manifest matching the 2024 schema.

WEEK-OFFSET VERIFICATION. The un-numbered = week 1 / (N) = week N+1 mapping is
inferred from download order, so it is checked before anything is written:
each source file's receiver list is matched (Jaccard) against nflverse
weekly receiving for the year+week it would map to, on several spot weeks. If
a spot week matches its own offset poorly AND matches some other offset
better, the whole reorg ABORTS with a report and touches nothing - a
mis-weeked defence archive silently poisons every backtest that reads it.

    python scripts/reorg_pff_weekly_archive.py --years 2022,2023
    python scripts/reorg_pff_weekly_archive.py --years 2022,2023 --dry-run
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import pandas as pd  # noqa: E402

PFF_ROOT = Path('pff_imports')
KINDS = ('receiving_summary', 'receiving_concept', 'receiving_scheme')
REG_WEEKS = 18
SPOT_WEEKS = (1, 5, 10, 15, 18)
MIN_JACCARD = 0.70          # a real week's PFF list vs nflverse targets>0
OFFSETS_TO_TRY = (1, 0, 2)  # candidate: file-index N -> week N+offset; un-numbered treated as index 0


def _src_dir(year):
    for name in ('Weekly Data', 'weekly_data', 'Weekly data'):
        d = PFF_ROOT / str(year) / name
        if d.is_dir():
            return d
    return None


def _index_of(fname):
    """'receiving_summary.csv' -> 0 ; 'receiving_summary (7).csv' -> 7."""
    m = re.search(r'\((\d+)\)', fname)
    return int(m.group(1)) if m else 0


def _receiver_set(path):
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    col = 'player' if 'player' in d.columns else d.columns[0]
    return {str(x).lower().strip() for x in d[col].dropna() if str(x).strip()}


def _nflverse_receivers(year, week):
    import nflreadpy as nfl
    w = nfl.load_player_stats(seasons=[year]).to_pandas()
    w = w[w['position'].isin(['WR', 'TE', 'RB']) & (w['week'] == week)]
    tgt = pd.to_numeric(w.get('targets', 0), errors='coerce').fillna(0)
    return {str(n).lower().strip() for n, t in zip(w['player_display_name'], tgt) if t > 0}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def verify_offset(year, src_dir):
    """Return the winning file-index -> week offset, or None if ambiguous."""
    summaries = {_index_of(p.name): p for p in src_dir.glob('receiving_summary*.csv')}
    print(f"\n{year}: {len(summaries)} summary files, indices "
          f"{min(summaries)}..{max(summaries)}")
    best_offset, best_score = None, -1.0
    detail = {}
    for off in OFFSETS_TO_TRY:
        scores = []
        for wk in SPOT_WEEKS:
            idx = wk - off
            if idx not in summaries:
                continue
            try:
                nv = _nflverse_receivers(year, wk)
            except Exception as exc:
                print(f"  nflverse fetch failed {year} wk{wk}: {exc}")
                return None
            j = _jaccard(_receiver_set(summaries[idx]), nv)
            scores.append(j)
        mean = sum(scores) / len(scores) if scores else 0.0
        detail[off] = mean
        print(f"  offset {off:+d} (file idx N -> week N+{off}): mean Jaccard {mean:.3f} "
              f"over {len(scores)} spot weeks")
        if mean > best_score:
            best_offset, best_score = off, mean
    if best_score < MIN_JACCARD:
        print(f"  ABORT {year}: best offset {best_offset:+d} only scores {best_score:.3f} "
              f"(< {MIN_JACCARD}). The week mapping is not trustworthy.")
        return None
    runner = sorted(v for k, v in detail.items() if k != best_offset)
    if runner and best_score - runner[-1] < 0.15:
        print(f"  ABORT {year}: offset {best_offset:+d} ({best_score:.3f}) not clearly ahead of "
              f"the next ({runner[-1]:.3f}) - ambiguous.")
        return None
    print(f"  {year}: offset {best_offset:+d} confirmed (Jaccard {best_score:.3f}).")
    return best_offset


def reorg_year(year, offset, dry_run):
    src_dir = _src_dir(year)
    week_root = PFF_ROOT / str(year) / 'weekly'
    idx_by_kind = {k: {_index_of(p.name): p for p in src_dir.glob(f'{k}*.csv')} for k in KINDS}
    written, manifest_rows = 0, []
    for idx in sorted(idx_by_kind['receiving_summary']):
        week = idx + offset
        if not (1 <= week <= REG_WEEKS):
            continue
        wk_dir = week_root / str(week)
        for kind in KINDS:
            src = idx_by_kind[kind].get(idx)
            if src is None:
                print(f"  {year} wk{week}: MISSING {kind}")
                continue
            dst = wk_dir / f'{kind}.csv'
            if dry_run:
                print(f"  would copy {src}  ->  {dst}")
            else:
                wk_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            written += 1
        manifest_rows.append(dict(week=week, regular_season=True, schema_valid=True,
                                  source_confidence=f'reorg_from_weekly_data_dump_offset{offset:+d}',
                                  export_date=pd.Timestamp.today().strftime('%Y-%m-%d')))
    if not dry_run and manifest_rows:
        week_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows).to_csv(week_root / 'manifest.csv', index=False)
    print(f"  {year}: {'(dry run) ' if dry_run else ''}{written} files, "
          f"{len(manifest_rows)} weeks, manifest {'—' if dry_run else week_root / 'manifest.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    years = [int(y) for y in a.years.replace(' ', '').split(',')]

    plan = {}
    for year in years:
        src = _src_dir(year)
        if src is None:
            print(f"{year}: no 'Weekly Data' folder under pff_imports/{year}/ - skipped")
            continue
        off = verify_offset(year, src)
        if off is None:
            raise SystemExit(f"\nweek-offset verification FAILED for {year}. Nothing written for any "
                             f"year. Fix the mapping or the source files and re-run.")
        plan[year] = off

    print(f"\n{'=' * 60}\nverification passed for {sorted(plan)}; writing\n{'=' * 60}")
    for year, off in plan.items():
        reorg_year(year, off, a.dry_run)


if __name__ == '__main__':
    main()
