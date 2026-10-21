"""
Server-side timing of Weekly Rankings interactions (2026-10-21, app speed): a full page run before any build, the
Build board click, a full page rerun with the board kept, the position-group buttons (fragment reruns), and a rerun
after the board's inputs change. AppTest runs the real script, so these are the seconds the browser waits for.

    python scripts/time_weekly_rankings.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402

from config import TAB_WEEKLY_FANTASY  # noqa: E402


def timed(label, fn):
    t = time.time()
    out = fn()
    print(f'{label:55s} {time.time() - t:6.1f}s', flush=True)
    return out


def main():
    at = AppTest.from_file('app.py', default_timeout=900)
    at.session_state['active_tab'] = TAB_WEEKLY_FANTASY
    at.session_state['weekly_fantasy_active_sub'] = 'Weekly Rankings'
    timed('full page run, no board yet', at.run)
    build = next(b for b in at.button if b.key == 'weekly_rank_build')
    timed('Build board click', lambda: build.click().run())
    errors = [e.value for e in at.error] + [str(e.message) for e in at.exception]
    if errors:
        print('errors:', errors[:3])
    timed('full page rerun with the board kept', at.run)
    for label in ('WR', 'RB', 'TE', 'QB', 'SUPERFLEX'):
        btns = [b for b in at.button if b.label == label]
        if btns:
            timed(f'position button {label} (fragment)', lambda b=btns[0]: b.click().run())
    timed('full page rerun again', at.run)


if __name__ == '__main__':
    main()
