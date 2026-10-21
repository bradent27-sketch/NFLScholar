"""
Headless audit of every app page (2026-10-21): renders each top-level tab and each Weekly Fantasy / Draft HQ sub-tab with
Streamlit's AppTest, on its default selections and with the season set to the current one, and reports every uncaught
exception, every error box (including the per-tab "hit an error" guard and its traceback), every warning, and the time
each render took. Nothing is clicked, so no board is built.

    python scripts/audit_app_pages.py [--season 2026] [--only "PLAYER SEARCH"] [--out .sweeps/app_audit.txt]
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402

from config import TAB_LABELS, TAB_WEEKLY_FANTASY  # noqa: E402

# the season selector key of each page (so a run can be repeated on the current season)
SEASON_KEYS = {'PLAYER SEARCH': 'year_tab1', 'NFL DEPTH CHARTS': 'year_tab2', 'DEFENSIVE YIELD SCHEMES': 'year_tab3',
               'PLAYER COMPARE': 'year_cmp', 'MATCHUP ANALYZER': 'ma_season', 'GAME SLATE': None,
               'Risers / Waiver Wire': 'year_tab4', 'Rookie Watch': 'year_tab5', 'Weekly Rankings': 'weekly_rank_year'}
WEEKLY_SUBS = ["Weekly Rankings", "Track Record", "Risers / Waiver Wire", "Rookie Watch"]


def texts(elements):
    out = []
    for e in elements:
        try:
            out.append(str(e.value)[:400])
        except Exception:
            out.append(repr(e)[:400])
    return out


def run_page(top, sub=None, season=None, timeout=900):
    at = AppTest.from_file('app.py', default_timeout=timeout)
    at.session_state['active_tab'] = top
    if sub:
        at.session_state['weekly_fantasy_active_sub'] = sub
    key = SEASON_KEYS.get(sub or top)
    if season is not None and key:
        at.session_state[key] = season
    t = time.time()
    try:
        at.run()
    except Exception:
        return {'crash': traceback.format_exc()[-1500:], 'seconds': time.time() - t}
    return {'seconds': time.time() - t,
            'exceptions': [str(getattr(e, 'message', e))[:600] + '\n' + '\n'.join(getattr(e, 'stack_trace', [])[-6:])
                           for e in at.exception],
            'errors': texts(at.error), 'warnings': texts(at.warning),
            'codes': [c.value[-1500:] for c in at.code if 'Traceback' in str(c.value)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, default=2026)
    ap.add_argument('--only', default=None)
    ap.add_argument('--out', default='.sweeps/app_audit.txt')
    a = ap.parse_args()
    pages = []
    for top in TAB_LABELS:
        if top == TAB_WEEKLY_FANTASY:
            pages += [(top, sub) for sub in WEEKLY_SUBS]
        else:
            pages.append((top, None))
    lines = []
    for top, sub in pages:
        label = f'{top} / {sub}' if sub else top
        if a.only and a.only not in label:
            continue
        for season in (None, a.season):
            key = SEASON_KEYS.get(sub or top)
            if season is not None and not key:
                continue
            res = run_page(top, sub, season)
            head = f"=== {label} | season {'default' if season is None else season} | {res['seconds']:.1f}s"
            print(head, flush=True)
            lines.append(head)
            if 'crash' in res:
                lines.append('  CRASH: ' + res['crash'])
                print('  CRASH', flush=True)
                continue
            for kind in ('exceptions', 'codes', 'errors', 'warnings'):
                for item in res[kind]:
                    lines.append(f'  {kind[:-1].upper()}: {item}')
            print(f"  exceptions {len(res['exceptions'])}, tracebacks {len(res['codes'])}, errors {len(res['errors'])}, "
                  f"warnings {len(res['warnings'])}", flush=True)
            with open(a.out, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
