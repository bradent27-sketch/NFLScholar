"""
The end-of-season routine (2026-11-11, audit D3, docs/audit_leads_2026-11-10.md). Run it once the season's week 17 is final.

Without --run it only checks and prints the plan. With --run it does the part that is a script today, in this order:
  1. scripts/deep_audit.py --stage all --years 2021..YEAR     rebuilds every board of today's model and the panel (~70 min)
  2. scripts/deep_audit.py --stage engines                     refits the prop engines on every panel season
  3. copies data/prop_engines.json to .sweeps/prop_engines_before_eos_<YEAR>.json   (the engines in use until now)
  4. scripts/prop_scorecard.py --install-engines               installs the refit engines
  5. scripts/eval_prop_engine_refit.py --install <stats>       re-installs the "mean": "board" engines for the stats that had
                                                               one before (step 4 would otherwise put those stats back on
                                                               the pre-recal column, the order the 2026-11-01 refit needs)
  6. scripts/prop_scorecard.py --year YEAR                     the season's prop record on the new engines
Every step logs to .sweeps/end_of_season_<YEAR>.log; it stops at the first failure. Steps 1-2 are heavy: launch the whole thing
detached through the queue, e.g. a .bat with `python scripts\\end_of_season.py --year 2026 --run` handed to
.sweeps\\wait_then_run.ps1 (the full run is over 2 hours, longer than a background command lives).

The rest of the routine is not a script yet; the plan lists it so it is not forgotten.

    python scripts/end_of_season.py --year 2026           # checks + plan
    python scripts/end_of_season.py --year 2026 --run     # runs steps 1-6
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENGINES = os.path.join(ROOT, 'data', 'prop_engines.json')
FIRST_PANEL_YEAR = 2021
LAST_SCORED_WEEK = 17          # week 18 is out of scope

MANUAL = """
Still by hand (docs/audit_leads_2026-11-10.md):
  - A10  re-earn ablation: rebuild the season with each family of flags shipped since September removed; report beside the
         harness deltas (nothing ships or retires from it alone).
  - B2/B8 refit the bet tiers' shrink and the stat switches on the full season, then freeze them for weeks 1-4 next season.
  - Week 1 before the opener: backlog M20 (rookie floor in backtests), M21 (week-1 QB1 from the archived chart), audit A4
         (rookie draft-capital prior), and store a pre-kickoff week-1 board.
  - Watch items W3-W6 and W9 with the new season added (docs/improvement_backlog.md section W).
  - D4: count the weekly PFF seasons on disk (scripts/pff_weekly_coverage.py); retry N2 / N5 / C1 / C1b once there are three.
Year-specific code to roll forward for a new season: scripts/deep_audit.py posted_lines_2026 (and its callers in
scripts/eval_prop_engine_refit.py), the default --years and stage_engines' `yr <= 2025` out-of-sample cut in
scripts/deep_audit.py, and DEFAULT_SHRINK in scripts/prop_scorecard.py once the tiers carry their own.
"""


def board_mean_stats(path=ENGINES):
    """The stats whose installed engine reads the board's column as is ("mean": "board")."""
    try:
        with open(path, encoding='utf-8') as fh:
            engines = json.load(fh)
    except (OSError, ValueError):
        return []
    return sorted(s for s, e in engines.items() if e.get('mean') == 'board')


def checks(year):
    """(ok, lines): the season's stats reach LAST_SCORED_WEEK, and the engine file is readable."""
    import logging
    logging.disable(logging.WARNING)
    import pandas as pd
    from data.transforms import load_and_merge_data
    lines, ok = [], True
    try:
        stats, _t, _n, _ = load_and_merge_data(year, 'Full PPR')
        last = int(pd.to_numeric(stats['week'], errors='coerce').max())
    except Exception as exc:
        last = 0
        lines.append(f'  stats for {year} could not be loaded: {exc}')
    ok &= last >= LAST_SCORED_WEEK
    lines.append(f'  {year} stats run through week {last}' + ('' if last >= LAST_SCORED_WEEK else
                                                              f' - wait for week {LAST_SCORED_WEEK} to be final'))
    board = board_mean_stats()
    ok &= bool(os.path.exists(ENGINES))
    lines.append(f'  engines: {ENGINES} ({"board-mean stats: " + ", ".join(board) if board else "none read as board mean"})')
    if year != 2026:
        lines.append('  NOTE: scripts/eval_prop_engine_refit.py scores 2026 posted lines only (posted_lines_2026); roll it forward')
    return ok, lines


def steps(year):
    years = ','.join(str(y) for y in range(FIRST_PANEL_YEAR, year + 1))
    backup = os.path.join(ROOT, '.sweeps', f'prop_engines_before_eos_{year}.json')
    board = board_mean_stats()
    py = sys.executable
    plan = [
        ('deep audit pass 1', [py, 'scripts/deep_audit.py', '--stage', 'all', '--years', years]),
        ('engine refit', [py, 'scripts/deep_audit.py', '--stage', 'engines', '--years', years]),
        ('back up the engines in use', ('copy', ENGINES, backup)),
        ('install the refit engines', [py, 'scripts/prop_scorecard.py', '--install-engines']),
    ]
    if board:
        plan.append(('re-install the board-mean engines', [py, 'scripts/eval_prop_engine_refit.py', '--install', ','.join(board)]))
    plan.append(('prop record on the new engines', [py, 'scripts/prop_scorecard.py', '--year', str(year),
                                                    '--out', f'.sweeps/prop_scorecard_eos_{year}.txt',
                                                    '--csv', f'.sweeps/prop_scorecard_eos_{year}_rows.csv']))
    return plan


def describe(step):
    name, cmd = step
    if isinstance(cmd, tuple):
        return f'{name}: copy {os.path.relpath(cmd[1], ROOT)} -> {os.path.relpath(cmd[2], ROOT)}'
    return f'{name}: ' + ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))


def run(plan, log_path):
    def log(msg):
        line = f'{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}'
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    for i, step in enumerate(plan, 1):
        name, cmd = step
        log(f'step {i}/{len(plan)} start - {describe(step)}')
        if isinstance(cmd, tuple):
            os.makedirs(os.path.dirname(cmd[2]), exist_ok=True)
            shutil.copy2(cmd[1], cmd[2])
            code = 0
        else:
            with open(log_path, 'a', encoding='utf-8') as fh:
                code = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if code != 0:
            log(f'step {i} FAILED (exit {code}); stopped. The engines in use are unchanged unless step 4 ran.')
            return code
        log(f'step {i} done')
    log('all steps done')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--run', action='store_true', help='run the steps (heavy; launch detached through the queue)')
    ap.add_argument('--force', action='store_true', help='run even when the checks fail (e.g. a partial season, for a test)')
    a = ap.parse_args()
    ok, lines = checks(a.year)
    plan = steps(a.year)
    print(f'End of season {a.year}\nChecks:\n' + '\n'.join(lines))
    print('Steps:\n' + '\n'.join(f'  {i}. {describe(s)}' for i, s in enumerate(plan, 1)))
    print(MANUAL)
    if not a.run:
        return 0
    if not ok and not a.force:
        print('Checks failed; not running (--force to override).')
        return 1
    return run(plan, os.path.join(ROOT, '.sweeps', f'end_of_season_{a.year}.log'))


if __name__ == '__main__':
    sys.exit(main())
