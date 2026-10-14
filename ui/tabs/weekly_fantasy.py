"""
Weekly Fantasy tab: Weekly Rankings, Track Record, Risers/Waiver Wire and
Rookie Watch, combined into one tab with sub-tabs - they answer some version
of "what should I do about my roster this week" (and, for Track Record, how
far to trust the board), which used to be spread across separate top-level
tabs. Track Record was added 2026-10-13.

No "WEEKLY FANTASY" heading here, same convention Draft HQ's own render()
uses for its sub-tabbed layout: the tab you clicked to get here is already
labelled that, and each sub-module below prints its own more specific
header anyway.

Sub-tabs are gated behind `.open`, same lazy-execution reasoning as
app.py's own top-level st.tabs (see its comment) - each of the three
sub-renderers does a real load_and_merge_data call plus its own board-build
function, and only one sub-tab is ever visible at a time.
"""
import streamlit as st

from ui.tabs import risers, rookie_watch, rankings, track_record

_SUB_LABELS = ["Weekly Rankings", "Track Record", "Risers / Waiver Wire", "Rookie Watch"]
_SUB_MODULES = [rankings, track_record, risers, rookie_watch]


def render():
    sub_tabs = st.tabs(_SUB_LABELS, key="weekly_fantasy_active_sub", on_change="rerun")
    for sub_tab, module in zip(sub_tabs, _SUB_MODULES):
        if sub_tab.open:
            with sub_tab:
                module.render()
