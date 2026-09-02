"""The weekly-rankings decomposition cache (`_selected_player_detail`).

Contract, per two explicit user asks that pull in opposite directions:

* a player's decomposition, once opened, stays memoized for the rest of the
  session - including across a plain board rebuild (TTL expiry, a week /
  scoring switch away and back); but
* a manual injury override / FantasyPros pull for the SAME week MUST bust it,
  for the injured player and for everyone the vacancy redistributes to,
  otherwise the table restates after "add injured player" while the open
  decomposition keeps showing the pre-injury breakdown (bug report
  2026-08-31).

The reconciliation: `detail_config`'s last element is this week's availability
fingerprint. A feed edit changes that element without touching the
(year, week, scoring) head, so it lands on a fresh cache key and the
stale-generation sweep drops the pre-injury copies for that same board.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui.tabs.rankings as rankings  # noqa: E402


def _use_fresh_session(monkeypatch):
    fake = {}
    monkeypatch.setattr(rankings.st, "session_state", fake)
    return fake


def test_decomposition_reloads_after_an_injury_fingerprint_change(monkeypatch):
    fake = _use_fresh_session(monkeypatch)
    key = ("Nico Collins", "WR", "HOU")
    meta_before = {"explanations": {key: {"player": "Nico Collins", "proj": 12.0}}}
    meta_after = {"explanations": {key: {"player": "Nico Collins", "proj": 9.5}}}
    cfg_before = (2026, 1, "Full PPR", "fp_before")
    cfg_after = (2026, 1, "Full PPR", "fp_after")

    assert rankings._selected_player_detail(meta_before, cfg_before, key)["proj"] == 12.0
    # Same config: served from the session cache even though meta changed under it.
    assert rankings._selected_player_detail(meta_after, cfg_before, key)["proj"] == 12.0
    # Injury edit -> fingerprint (last element) moves -> forced reload from the rebuild.
    assert rankings._selected_player_detail(meta_after, cfg_after, key)["proj"] == 9.5

    cache = fake[rankings._PLAYER_DETAIL_CACHE_KEY]
    assert (cfg_before, key) not in cache  # stale generation pruned
    assert (cfg_after, key) in cache


def test_decomposition_survives_a_same_fingerprint_rebuild(monkeypatch):
    _use_fresh_session(monkeypatch)
    key = ("Derrick Henry", "RB", "BAL")
    cfg = (2026, 1, "Full PPR", "fp_stable")
    rankings._selected_player_detail({"explanations": {key: {"proj": 20.0}}}, cfg, key)
    # A week switch away and back recomputes the SAME fingerprint: still cached,
    # so a fresh deep copy of model_meta['explanations'] is not re-walked.
    kept = rankings._selected_player_detail({"explanations": {key: {"proj": 999.0}}}, cfg, key)
    assert kept["proj"] == 20.0


def test_stale_sweep_only_touches_the_edited_board(monkeypatch):
    fake = _use_fresh_session(monkeypatch)
    key = ("Player A", "WR", "X")
    wk1 = (2026, 1, "Full PPR", "fpA")
    wk2 = (2026, 2, "Full PPR", "fpB")
    rankings._selected_player_detail({"explanations": {key: {"v": 1}}}, wk1, key)
    rankings._selected_player_detail({"explanations": {key: {"v": 2}}}, wk2, key)

    # An injury edit on week 1 only.
    wk1_edited = (2026, 1, "Full PPR", "fpA2")
    rankings._selected_player_detail({"explanations": {key: {"v": 3}}}, wk1_edited, key)

    cache = fake[rankings._PLAYER_DETAIL_CACHE_KEY]
    assert (wk1, key) not in cache       # stale week-1 generation swept
    assert (wk2, key) in cache           # week-2 entry untouched
    assert (wk1_edited, key) in cache


def test_missing_explanation_is_not_cached(monkeypatch):
    fake = _use_fresh_session(monkeypatch)
    key = ("Nobody", "WR", "ZZ")
    cfg = (2026, 1, "Full PPR", "fp")
    assert rankings._selected_player_detail({"explanations": {}}, cfg, key) is None
    assert (cfg, key) not in fake.get(rankings._PLAYER_DETAIL_CACHE_KEY, {})
