"""
NFL Scholar - entrypoint. Page config, theme injection, sidebar diagnostics,
and tab wiring only; all data/UI logic lives in data/ and ui/.
"""
import traceback

import streamlit as st

from config import TAB_LABELS
from ui.styling import inject_theme
from ui.components import render_data_health_sidebar, ensure_pff_imports_dir, render_intro_and_glossary
from ui.tabs import (
    player_search, depth_charts, defensive_yield, risers, rookie_watch, rankings, vorp_draft, live_odds, player_compare,
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
# of all 8 tabs' full data-load-and-render pipeline running on every single
# widget interaction anywhere in the app, which was the actual root cause
# behind Rookie Watch (and everything else) feeling sluggish.
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")

_tab_modules = [player_search, depth_charts, defensive_yield, risers, rookie_watch, rankings, live_odds, player_compare, vorp_draft]
for _tab, _module, _label in zip([tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9], _tab_modules, TAB_LABELS):
    if _tab.open:
        with _tab:
            _render_guarded(_module, _label)
