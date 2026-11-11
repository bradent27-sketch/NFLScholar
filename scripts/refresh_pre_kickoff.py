"""
Pre-kickoff refresh (2026-11-11, audit B10, docs/audit_leads_2026-11-10.md), run by Task Scheduler about 75 minutes before each
kickoff window (scripts/schedule_tasks.ps1). The ledger's last pre-kickoff board is what the track record scores, and it used
to be whatever was built by hand, sometimes days before the game. This writes a fresh one from the latest news and lines:

  1. skips (and logs why) when a heavy job is running on the shared machine or free memory is under 4 GB
     (the same gate as .sweeps/wait_then_run.ps1);
  2. the FantasyPros injury pull for the week, when a key is saved (1 call of the 500/day cap);
  3. a forced pull of the free weekly books (data.odds_weekly.weekly_props, archived to external_data/props_archive);
  4. the board, built exactly as the app's Build board does (build_weekly_projections with board_inputs' fingerprint,
     Full PPR), the market frame and prop history as the app builds them, the FantasyPros weekly columns carried over from the
     week's previous ledger board (a new FP weekly pull costs 4 calls and is left to the app);
  5. one ledger board (record_board; its one-per-hour throttle still holds) and .sweeps/scheduled/what_moved_wkNN.md: players
     whose points moved 1+ or a main stat 10%+ since the week's previous board.

It never touches the app's kept board (ui/board_pin). No market number moves a projection: the market frame is stored beside
the board, as the app does.

    python scripts/refresh_pre_kickoff.py [--year 2026 --week 10] [--no-gate]
"""
import argparse
import datetime
import glob
import json
import logging
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
for _stream in ('stdout', 'stderr'):     # pythonw (Task Scheduler) has no console
    if getattr(sys, _stream) is None:
        setattr(sys, _stream, open(os.devnull, 'w'))
logging.disable(logging.WARNING)

import pandas as pd  # noqa: E402

LOG = os.path.join(ROOT, '.sweeps', 'scheduled', 'refresh_pre_kickoff.log')
HEAVY = (r'week_review[\\/]|ablation_pairs\.py|flag_pairs\.py|backtest_component\.py|run_guarded_harness\.py|'
         r'capture_stat_blend_traces\.py|deep_audit\.py')
MIN_FREE_GB = 4.0
SCORING = 'Full PPR'
MOVE_PTS, MOVE_STAT = 1.0, 0.10
MOVE_STATS = ('passing_yards', 'rushing_yards', 'receptions', 'receiving_yards', 'targets', 'rushing_attempts')


def log(msg):
    line = f'{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}'
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def machine_busy():
    """A reason to skip, or None: a heavy job running (another session's), or less than MIN_FREE_GB of free commit."""
    ps = ("$h = Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match '" + HEAVY + "' } | "
          "Measure-Object | Select-Object -ExpandProperty Count; "
          "$f = (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory; "
          "@{heavy=$h; free_kb=$f} | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], capture_output=True,
                             text=True, timeout=60)
        info = json.loads(out.stdout.strip() or '{}')
    except Exception as exc:
        return None if os.name != 'nt' else f'machine check failed ({type(exc).__name__})'
    heavy, free_gb = int(info.get('heavy') or 0), float(info.get('free_kb') or 0) / 1024 / 1024
    if heavy:
        return f'{heavy} heavy job(s) running'
    if free_gb < MIN_FREE_GB:
        return f'{free_gb:.1f} GB free (needs {MIN_FREE_GB:.0f})'
    return None


def target_week(now=None):
    """(season, week) of the coming slate, from the schedule (data.odds_weekly._season_week_for_props), on the local date: a
    Monday-night refresh is already Tuesday in UTC."""
    from data.odds_weekly import _season_week_for_props
    return _season_week_for_props(now or datetime.datetime.now())


def previous_board(year, week):
    paths = sorted(glob.glob(os.path.join('data', 'ledger', f'{year}_wk{week:02d}_*.parquet')), key=os.path.getmtime)
    for path in reversed(paths):
        try:
            return path, pd.read_parquet(path)
        except Exception:
            continue
    return None, None


def fp_weekly_from(board):
    """The FantasyPros weekly frame the ledger stores, rebuilt from a ledger board's FP columns (None without them)."""
    if board is None or 'Player' not in board.columns:
        return None
    pts = [c for c in board.columns if c.startswith('FP Proj Pts')]
    stats = [c for c in board.columns if c.startswith('FP ') and c not in pts]
    if not pts and not stats:
        return None
    fp = board[['Player'] + pts + stats].rename(columns={c: c[3:] for c in stats})
    keep = fp[pts + [c[3:] for c in stats]].notna().any(axis=1)
    return fp[keep].reset_index(drop=True) if keep.any() else None


def what_moved(before, after):
    """Players whose points moved MOVE_PTS+ or a main stat MOVE_STAT+ (relative, on a base of 5+), with the availability or
    injury status when that changed: a markdown table, or '' when nothing moved."""
    if before is None or after is None:
        return ''
    pts = 'Raw Model Proj Pts'
    cols = ['Player', 'Pos', 'Team', pts, 'Availability', 'Injury Status'] + [s for s in MOVE_STATS]
    b = before[[c for c in cols if c in before.columns]].drop_duplicates('Player').set_index('Player')
    a = after[[c for c in cols if c in after.columns]].drop_duplicates('Player').set_index('Player')
    both = a.index.intersection(b.index)
    rows = []
    for p in both:
        why = []
        if pts in a.columns and pts in b.columns:
            d = float(a.at[p, pts]) - float(b.at[p, pts])
            if abs(d) >= MOVE_PTS:
                why.append(f'pts {float(b.at[p, pts]):.1f} -> {float(a.at[p, pts]):.1f}')
        for s in MOVE_STATS:
            if s in a.columns and s in b.columns:
                x, y = float(b.at[p, s] or 0), float(a.at[p, s] or 0)
                if max(x, y) >= 5 and abs(y - x) / max(x, 1e-9) >= MOVE_STAT:
                    why.append(f'{s} {x:.1f} -> {y:.1f}')
        if not why:
            continue
        status = []
        for c in ('Availability', 'Injury Status'):
            if c in a.columns and c in b.columns and str(a.at[p, c]) != str(b.at[p, c]):
                status.append(f'{c.lower()} {b.at[p, c]} -> {a.at[p, c]}')
        rows.append((p, a.at[p, 'Pos'] if 'Pos' in a.columns else '', a.at[p, 'Team'] if 'Team' in a.columns else '',
                     '; '.join(why), '; '.join(status) or 'own status unchanged (a teammate, a QB1 choice or a new line)'))
    gone = sorted(set(b.index) - set(a.index))
    new = sorted(set(a.index) - set(b.index))
    if not rows and not gone and not new:
        return ''
    out = ['| Player | Pos | Team | Moved | Why |', '|---|---|---|---|---|']
    out += [f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |' for r in sorted(rows, key=lambda r: (r[2], r[0]))]
    if new:
        out += ['', f'New on the board: {", ".join(new[:40])}' + (' ...' if len(new) > 40 else '')]
    if gone:
        out += ['', f'Off the board: {", ".join(gone[:40])}' + (' ...' if len(gone) > 40 else '')]
    return '\n'.join(out)


def injury_pull(year, week):
    try:
        from data.draft_sources import get_fantasypros_api_key
        key, _src = get_fantasypros_api_key()
    except Exception:
        key = None
    if not key:
        return 'FantasyPros injury pull skipped (no key)'
    from data.fantasypros_availability import save_fantasypros_injury_api_pull
    n, err = save_fantasypros_injury_api_pull(key, year, week)
    return f'FantasyPros injury pull: {err}' if err else f'FantasyPros injury pull: {n} rows'


def refresh(year, week):
    from data.draft_board import DEFAULT_SCORING
    from data.odds_weekly import weekly_market_frame, weekly_props
    from data.prediction_ledger import record_board
    from data.prop_probability import player_history
    from data.transforms import load_and_merge_data
    from data.weekly_projections import DEFAULT_FEATURES, build_weekly_projections
    from ui.board_session import board_inputs

    t0 = time.time()
    log(injury_pull(year, week))
    props, meta = weekly_props(force=True)
    log(f"props pull: {len(props)} rows{' (stale)' if meta.get('stale') else ''}")
    board, _meta = build_weekly_projections(year, week, SCORING, availability_fingerprint=board_inputs(year, week)[0])
    if board is None or board.empty:
        log(f'{year} week {week}: the board came back empty; nothing recorded')
        return 1
    scoring = {**DEFAULT_SCORING, 'rec': 1.0}
    market = weekly_market_frame(props, scoring, name_pool=board[['Player', 'Pos']], fetched_at=meta.get('fetched_at'))
    stats_df, _tc, name_col, _ = load_and_merge_data(year, SCORING)
    try:
        prior, _pt, prior_name_col, _ = load_and_merge_data(year - 1, SCORING)
    except Exception:
        prior, prior_name_col = None, None
    history = player_history(stats_df, name_col, int(week), prior, prior_name_col) if len(market) else None
    prev_path, prev = previous_board(year, week)
    path = record_board(year, week, board, market_df=market if len(market) else None, fp_weekly=fp_weekly_from(prev),
                        feats=DEFAULT_FEATURES, scoring=SCORING, history=history)
    if path and path == prev_path:
        log(f'{year} week {week}: a board from under an hour ago already stands ({os.path.basename(path)}); not rewritten')
        return 0
    log(f"{year} week {week}: {len(board)} players, {len(market)} priced, ledger {os.path.basename(path) if path else 'NOT written'}"
        f", {time.time() - t0:.0f}s")
    if path:
        moved = what_moved(prev, pd.read_parquet(path))
        note = os.path.join('.sweeps', 'scheduled', f'what_moved_{year}_wk{week:02d}.md')
        with open(note, 'a', encoding='utf-8') as fh:
            fh.write(f"\n## {datetime.datetime.now():%a %Y-%m-%d %H:%M} vs {os.path.basename(prev_path) if prev_path else '-'}\n\n"
                     + (moved or 'Nothing moved past the thresholds.') + '\n')
        log(f'what moved -> {note}')
    return 0 if path else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int)
    ap.add_argument('--week', type=int)
    ap.add_argument('--no-gate', action='store_true', help='skip the heavy-job / memory check')
    a = ap.parse_args()
    if not a.no_gate:
        busy = machine_busy()
        if busy:
            log(f'skipped: {busy}')
            return 0
    year, week = (a.year, a.week) if a.year and a.week else target_week()
    if not year or not week:
        log('skipped: no upcoming week on the schedule')
        return 0
    try:
        return refresh(int(year), int(week))
    except Exception as exc:
        log(f'failed: {type(exc).__name__}: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
