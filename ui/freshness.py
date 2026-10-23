"""Freshness notes for the Weekly Fantasy board (backlog D2, 2026-10-19).

Week 5: Chase and McLaurin were listed Out on the last pre-kickoff board and played (19.7 and 12.2 points). The Out
status was real when the injury report was pulled; the inactives list posts about 90 minutes before each kickoff and the
board was not rebuilt after it. This module turns the injury file age and the next kickoff into one short note beside
the Build board button. Pure functions, no Streamlit, so they are unit-tested.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
INACTIVES_LEAD = dt.timedelta(minutes=90)      # inactives post about 90 minutes before kickoff
NUDGE_WINDOW = dt.timedelta(hours=5)           # only nag when a kickoff is this close


def _span(delta: dt.timedelta) -> str:
    minutes = max(int(delta.total_seconds() // 60), 0)
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def span_text(delta: dt.timedelta) -> str:
    """'2h 05m' / '3d 4h' / '12m' - the same wording as the notes, for a label elsewhere."""
    return _span(delta)


def next_kickoff(schedule, week: int, now: dt.datetime) -> dt.datetime | None:
    """The earliest kickoff in `week` that is still ahead of `now` (timezone-aware, US Eastern), or None."""
    if schedule is None or len(schedule) == 0 or "gameday" not in schedule.columns:
        return None
    rows = schedule[schedule["week"] == week] if "week" in schedule.columns else schedule
    best = None
    for _, g in rows.iterrows():
        try:
            day = dt.date.fromisoformat(str(g.get("gameday"))[:10])
            hh, mm = (int(x) for x in str(g.get("gametime") or "13:00").split(":")[:2])
        except (ValueError, TypeError):
            continue
        ko = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=EASTERN)
        if ko > now and (best is None or ko < best):
            best = ko
    return best


def injury_report_note(saved_at: dt.datetime | None, kickoff: dt.datetime | None, now: dt.datetime):
    """(text, level) where level is "warn" or "info". `saved_at`, `kickoff` and `now` must all be timezone-aware."""
    if saved_at is None:
        saved_txt = "no injury report saved yet"
    else:
        saved_txt = f"injury report saved {_span(now - saved_at)} ago"
    if kickoff is None:
        return f"🩹 {saved_txt[0].upper() + saved_txt[1:]}.", "info"
    until = kickoff - now
    inactives_at = kickoff - INACTIVES_LEAD
    stale = saved_at is None or saved_at < inactives_at
    if until <= NUDGE_WINDOW and stale:
        return (f"🩹 Next kickoff in {_span(until)}, inactives post about 90 minutes before it, and the {saved_txt}. "
                "Pull the injury report again and rebuild once inactives are out so late Outs are caught.", "warn")
    return (f"🩹 {saved_txt[0].upper() + saved_txt[1:]}. Next kickoff in {_span(until)}; pull it again about 90 minutes "
            "before each kickoff for the inactives.", "info")
