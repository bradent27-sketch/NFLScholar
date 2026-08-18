"""
Weekly Fantasy tab: Risers/Waiver Wire, Rookie Watch and Weekly Rankings,
combined into one tab with sub-tabs - all three answer some version of
"what should I do about my roster this week", which used to be spread
across three separate top-level tabs.

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

from ui.tabs import risers, rookie_watch, rankings

_SUB_LABELS = ["Weekly Rankings", "Risers / Waiver Wire", "Rookie Watch"]
_SUB_MODULES = [rankings, risers, rookie_watch]


def render():
    sub_tabs = st.tabs(_SUB_LABELS, key="weekly_fantasy_active_sub", on_change="rerun")
    for sub_tab, module in zip(sub_tabs, _SUB_MODULES):
        if sub_tab.open:
            with sub_tab:
                module.render()
