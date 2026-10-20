"""ui/freshness.py: the injury-report freshness note and the next-kickoff lookup (backlog D2)."""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import freshness as fr  # noqa: E402

ET = fr.EASTERN


def at(day, hh, mm=0):
    return dt.datetime(2026, 10, day, hh, mm, tzinfo=ET)


def test_next_kickoff_is_the_earliest_future_game_of_the_week():
    sch = pd.DataFrame([
        {"week": 8, "gameday": "2026-10-22", "gametime": "20:15"},
        {"week": 8, "gameday": "2026-10-25", "gametime": "13:00"},
        {"week": 8, "gameday": "2026-10-25", "gametime": "16:25"},
        {"week": 9, "gameday": "2026-11-01", "gametime": "13:00"},
    ])
    assert fr.next_kickoff(sch, 8, at(20, 9)) == at(22, 20, 15)
    assert fr.next_kickoff(sch, 8, at(23, 9)) == at(25, 13)
    assert fr.next_kickoff(sch, 8, at(26, 9)) is None
    assert fr.next_kickoff(pd.DataFrame(), 8, at(20, 9)) is None


def test_next_kickoff_skips_a_game_with_no_usable_date():
    sch = pd.DataFrame([{"week": 8, "gameday": None, "gametime": "13:00"},
                        {"week": 8, "gameday": "2026-10-25", "gametime": "13:00"}])
    assert fr.next_kickoff(sch, 8, at(20, 9)) == at(25, 13)


def test_stale_file_close_to_kickoff_warns():
    text, level = fr.injury_report_note(saved_at=at(25, 8), kickoff=at(25, 13), now=at(25, 11))
    assert level == "warn" and "Next kickoff in 2h 00m" in text and "rebuild" in text


def test_file_saved_after_the_inactives_window_does_not_warn():
    # kickoff 13:00 -> inactives about 11:30; a file saved 11:45 already has them
    text, level = fr.injury_report_note(saved_at=at(25, 11, 45), kickoff=at(25, 13), now=at(25, 12))
    assert level == "info"


def test_a_distant_kickoff_is_information_only():
    text, level = fr.injury_report_note(saved_at=at(20, 9), kickoff=at(25, 13), now=at(21, 9))
    assert level == "info" and "Next kickoff in 4d 4h" in text


def test_missing_file_close_to_kickoff_warns_and_no_kickoff_is_quiet():
    text, level = fr.injury_report_note(saved_at=None, kickoff=at(25, 13), now=at(25, 10))
    assert level == "warn" and "no injury report saved" in text
    text, level = fr.injury_report_note(saved_at=at(24, 9), kickoff=None, now=at(25, 10))
    assert level == "info" and text.startswith("🩹 Injury report saved 1d")
