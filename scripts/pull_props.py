"""
Headless weekly props pull (2026-11-11, audit B6, docs/audit_leads_2026-11-10.md), for the scheduled pulls set up with
scripts/schedule_tasks.ps1. One forced pull of the free weekly books (PrizePicks, Underdog, DraftKings, Pinnacle, FanDuel:
data.odds_weekly.WEEKLY_BOOKS, no key and no quota), saved to the app's snapshot and archived to external_data/props_archive -
the same thing the app's Market props refresh does. Regular pulls give the record a real open and close instead of whatever was
pulled by hand.

--odds-api adds The Odds API's event props for the other US books (line shopping, B4). That one costs credits: it runs only with
a saved key, only for the stats the bet tiers use, and only when the account keeps --reserve credits after it
(data.odds_api_props). Off unless asked.

    python scripts/pull_props.py [--odds-api [--reserve 100]]
"""
import argparse
import datetime
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
for _stream in ('stdout', 'stderr'):     # pythonw (Task Scheduler) has no console
    if getattr(sys, _stream) is None:
        setattr(sys, _stream, open(os.devnull, 'w'))
logging.disable(logging.WARNING)

LOG = os.path.join(ROOT, '.sweeps', 'scheduled', 'pull_props.log')


def log(msg):
    line = f'{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}'
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def describe_status(status, now=None):
    """One short line per book: rows, live or saved file (with its age)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    parts = []
    for book, s in (status or {}).items():
        if not isinstance(s, dict):
            continue
        src = s.get('source') or '-'
        if s.get('saved_at'):
            try:
                age = now - datetime.datetime.fromisoformat(s['saved_at'])
                src = f'saved file {age.total_seconds() / 3600:.0f}h old'
            except ValueError:
                src = 'saved file'
        parts.append(f"{book} {s.get('rows', 0)} ({src}{'; ' + str(s['error'])[:60] if s.get('error') else ''})")
    return '; '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--odds-api', action='store_true', help="also pull The Odds API's event props (costs credits)")
    ap.add_argument('--reserve', type=int, default=None, help='credits the Odds API pull must leave unspent')
    a = ap.parse_args()
    from data.odds_weekly import weekly_props
    props, meta = weekly_props(force=True)
    week = None
    try:
        from data.odds_weekly import _season_week_for_props
        week = _season_week_for_props(datetime.datetime.now())
    except Exception:
        pass
    log(f"props pull: {len(props)} rows{', stale (nothing new came back)' if meta.get('stale') else ''}"
        f"{f', season/week {week}' if week else ''}; {describe_status(meta.get('status'))}"
        + (f"; SAVE ERROR {meta['save_error']}" if meta.get('save_error') else ''))
    code = 0 if len(props) and not meta.get('stale') else 1
    if a.odds_api:
        from data import odds_api_props
        kwargs = {} if a.reserve is None else {'reserve': a.reserve}
        result = odds_api_props.pull_and_archive(**kwargs)
        log(f"odds api: {result}")
    return code


if __name__ == '__main__':
    sys.exit(main())
