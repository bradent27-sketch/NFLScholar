"""
NFL Scholar - entrypoint. Page config, theme injection, sidebar diagnostics,
and tab wiring only; all data/UI logic lives in data/ and ui/.
"""
import traceback

import pandas as pd
import streamlit as st

# Keep pandas' string columns on numpy objects rather than Arrow.
#
# THIS PREVENTS HARD CRASHES, not a deprecation warning. pandas 3 stores
# strings in Arrow by default when pyarrow is installed, and on this stack
# (pandas 3.0.5 / pyarrow 25.0.0) that combination segfaults - the whole
# interpreter dies, with no traceback and no Streamlit error page, so the app
# just vanishes mid-draft. It was reproduced repeatedly a few picks into a
# session and traced with faulthandler into pyarrow: first in `compute.take`
# during a boolean row selection, then, once that call site was worked
# around, in `string_arrow._from_sequence` during a plain `.astype(str)`.
#
# Chasing it call site by call site was the wrong shape of fix - the hazard
# is every string operation in the process, and this app does thousands. One
# option at startup moves all of them onto the numpy path, which is where
# they were before pandas 3 and is unaffected.
#
# Set before any module builds a DataFrame, hence its position above the
# project imports.
pd.options.mode.string_storage = "python"

from config import TAB_LABELS  # noqa: E402
from ui.styling import inject_theme  # noqa: E402
from ui.components import (  # noqa: E402
    render_data_health_sidebar, ensure_pff_imports_dir, render_intro_and_glossary,
)
from ui.tabs import (  # noqa: E402
    game_slate, player_search, depth_charts, defensive_yield, risers, rookie_watch, rankings, live_odds,
    player_compare, matchup_analyzer, draft_hq,
)


def _render_guarded(tab_module, tab_label):
    """
    One tab blowing up should degrade to an error message inside THAT tab,
    never a full-page raw traceback taking the whole app down - during a
    real season the most likely failure is one source's schema drifting
    (a fresh weekly CSV drop, an nflverse column rename, a PFF export
    variation), and every other tab's data is still perfectly good when
    that happens. The traceback is kept, but inside an expander, so
    diagnosing is still one click - not hidden, just not the whole page.
    """
    try:
        tab_module.render()
    except Exception:
        st.error(
            f"The {tab_label} tab hit an error and couldn't finish rendering. "
            "The other tabs are unaffected - this usually means one data file "
            "changed shape (a fresh export with different columns) or a "
            "network source was unreachable."
        )
        with st.expander("Technical details (for debugging)"):
            st.code(traceback.format_exc())

st.set_page_config(page_title="NFL Scholar", layout="wide", page_icon="🏈")
inject_theme()
ensure_pff_imports_dir()

render_data_health_sidebar()
render_intro_and_glossary()

# key= + on_change="rerun" makes the active tab live in
# st.session_state["active_tab"] (Streamlit >=1.59), which is what lets
# ui.components.switch_tab() jump here from another tab (e.g. clicking a
# player row on Risers). It also makes each tab's .open property real
# (True only for the active tab) instead of always None - every tab below
# is now skipped entirely unless it's the one currently showing, instead
# of every tab's full data-load-and-render pipeline running on every single
# widget interaction anywhere in the app, which was the actual root cause
# behind Rookie Watch (and everything else) feeling sluggish.
#
# Unpacked into a list rather than N named variables so adding a tab is a
# two-line change (a label in config.TAB_LABELS, a module here) instead of
# also renumbering a tab1..tabN tuple that has to stay exactly as long as
# TAB_LABELS - a mismatch there fails with an opaque unpacking error rather
# than pointing at the tab that was actually added.
_tabs = st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")

_tab_modules = [game_slate, player_search, depth_charts, defensive_yield, risers, rookie_watch, rankings, live_odds, player_compare, matchup_analyzer, draft_hq]
assert len(_tab_modules) == len(TAB_LABELS), "TAB_LABELS and _tab_modules must stay in sync"
for _tab, _module, _label in zip(_tabs, _tab_modules, TAB_LABELS):
    if _tab.open:
        with _tab:
            _render_guarded(_module, _label)
