"""Focused offline tests for Ourlads fullback and literal-rank evidence."""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.ourlads_depth_charts as odc  # noqa: E402


def _chart(team: str = "MIA", name: str = "Miami Dolphins") -> bytes:
    """A minimal saved printer-friendly page. `team`/`name` let a test build
    a second, DIFFERENT team's page for the batched-import case."""
    return _CHART_TEMPLATE.replace(b"MIA", team.encode()).replace(
        b"Miami Dolphins", name.encode())


_CHART_TEMPLATE = b'''From: <Saved by Blink>
MIME-Version: 1.0
Content-Type: multipart/related; boundary="chart"

--chart
Content-Type: text/html; charset=UTF-8
Content-Location: https://www.ourlads.com/nfldepthcharts/pfdepthchart/MIA

<html><head><title>Miami Dolphins Depth Chart</title></head><body>
<table><tr><th>Pos</th><th>Player 1</th><th>Player 2</th><th>Player 3</th></tr>
<tr><td>RB</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/1/">Back, Missing 26/1</a></td><td><a href="https://www.ourlads.com/nfldepthcharts/player/2/">Back, Known 26/1</a></td><td><a href="https://www.ourlads.com/nfldepthcharts/player/3/" class="lc_red">Back, Third 26/1</a></td></tr>
<tr><td>FB</td><td><a href="https://www.ourlads.com/nfldepthcharts/player/4/">Ingold, Alec 20/3</a></td><td><a href="https://www.ourlads.com/nfldepthcharts/player/5/">Herman, Dj 25/7</a></td><td></td></tr>
</table></body></html>
--chart--
'''


class OurladsFullbackAndRankTests(unittest.TestCase):
    def setUp(self):
        self.snapshot, report = odc.parse_ourlads_depth_chart(_chart(), "Miami.mhtml")
        self.assertEqual(report["error"], "")
        self.roster = pd.DataFrame([
            {"name": "Known Back", "team": "MIA", "position": "RB"},
            {"name": "Third Back", "team": "MIA", "position": "RB"},
            # nflverse groups a fullback under RB; the narrow roster field
            # must keep him out of the core-RB Ourlads signal.
            {"name": "Alec Ingold", "team": "MIA", "position": "RB", "depth_chart_position": "FB"},
            # A real 2026-08-24 miscall: the roster's OWN depth_chart_position
            # is a stale "RB" (nflverse never granularly tagged this backup
            # fullback), while Ourlads' current, curated chart specifically
            # lists him at FB2. The old precedence trusted the roster field
            # unconditionally and let him compete for real core-RB volume.
            {"name": "Dj Herman", "team": "MIA", "position": "RB", "depth_chart_position": "RB"},
        ])

    def test_parser_retains_fullbacks(self):
        ingold = self.snapshot.loc[self.snapshot["player"].eq("Alec Ingold")].iloc[0]
        self.assertEqual(ingold["position"], "FB")
        self.assertTrue(ingold["is_listed_starter"])

    def test_unmatched_intermediate_player_never_renumbers_literal_rank(self):
        signal = odc.build_ourlads_projection_signal(self.snapshot, self.roster, "name", "team")
        known = signal["matches"].loc[signal["matches"]["matched_player"].eq("Known Back")].iloc[0]
        self.assertEqual(known["source_depth_rank"], 2)
        self.assertEqual(known["source_rank"], 2)

        source_rbs = signal["source_roles"].loc[
            signal["source_roles"]["source_position"].eq("RB")
        ].sort_values("source_rank")
        self.assertEqual(source_rbs["source_rank"].tolist(), [1, 2, 3])
        self.assertEqual(source_rbs["match_status"].tolist(), ["unmatched", "matched", "matched"])
        self.assertEqual(source_rbs.iloc[2]["source_status"], "inactive")

    def test_fullback_uses_narrow_roster_role_and_stays_out_of_legacy_skill_roles(self):
        signal = odc.build_ourlads_projection_signal(self.snapshot, self.roster, "name", "team")
        fullback = signal["fullback_roles"].iloc[0]
        self.assertEqual(fullback["matched_player"], "Alec Ingold")
        self.assertEqual(fullback["position"], "FB")
        self.assertEqual(fullback["source_position"], "FB")
        self.assertEqual(fullback["roster_position"], "RB")
        self.assertEqual(fullback["roster_depth_chart_position"], "FB")
        self.assertNotIn("Alec Ingold", set(signal["skill_roles"]["matched_player"]))

    def test_roster_fullback_role_overrides_a_coarse_source_rb_label(self):
        coarse = self.snapshot.copy()
        coarse.loc[coarse["player"].eq("Alec Ingold"), ["position", "position_label"]] = "RB"
        signal = odc.build_ourlads_projection_signal(coarse, self.roster, "name", "team")
        fullback = signal["fullback_roles"].iloc[0]
        self.assertEqual(fullback["source_position"], "RB")
        self.assertEqual(fullback["functional_position"], "FB")

    def test_ourlads_fullback_listing_overrides_a_stale_roster_rb_depth_chart_position(self):
        signal = odc.build_ourlads_projection_signal(self.snapshot, self.roster, "name", "team")
        herman = signal["matches"].loc[signal["matches"]["matched_player"].eq("Dj Herman")].iloc[0]
        self.assertEqual(herman["source_position"], "FB")
        self.assertEqual(herman["roster_depth_chart_position"], "RB")
        self.assertEqual(herman["functional_position"], "FB")
        self.assertNotIn("Dj Herman", set(signal["skill_roles"]["matched_player"]))
        fullbacks = set(signal["fullback_roles"]["matched_player"])
        self.assertIn("Dj Herman", fullbacks)
        self.assertIn("Alec Ingold", fullbacks)

    def test_reviewed_name_alias_bridges_source_only_spelling_difference(self):
        chart = self.snapshot.copy()
        mask = chart["player"].eq("Known Back")
        chart.loc[mask, "player"] = "Kenny Gainwell"
        chart.loc[mask, "player_key"] = odc.clean_name_exact(pd.Series(["Kenny Gainwell"])).iloc[0]
        roster = pd.DataFrame([
            {"name": "Kenneth Gainwell", "team": "MIA", "position": "RB"},
        ])
        signal = odc.build_ourlads_projection_signal(chart, roster, "name", "team")
        matched = signal["matches"].loc[
            signal["matches"]["matched_player"].eq("Kenneth Gainwell")
        ].iloc[0]
        self.assertEqual(matched["match_method"], "reviewed alias")
        self.assertEqual(matched["source_rank"], 2)


_MULTI_UNIT_PAGE = b"""From: <Saved by Blink>
MIME-Version: 1.0
Content-Type: multipart/related; boundary="chart"

--chart
Content-Type: text/html; charset=UTF-8
Content-Location: https://www.ourlads.com/nfldepthcharts/pfdepthchart/SEA

<html><head><title>Seattle Seahawks Depth Chart</title></head><body>
<table><tr><th>Pos</th><th>Player 1</th><th>Player 2</th></tr>
<tr><td>LWR</td><td>Real, Starter 24/1</td><td>Real, Backup 25/2</td></tr>
<tr><td>QB</td><td>Real, Passer 22/1</td><td></td></tr>
<tr><td>RB</td><td>Real, Runner 23/1</td><td></td></tr>
</table>
<table><tr><th>Pos</th><th>Player 1</th><th>Player 2</th></tr>
<tr><td>LDE</td><td>Def, End 24/1</td><td>Def, Two 25/1</td></tr>
<tr><td>NT</td><td>Def, Tackle 24/1</td><td>Def, Four 25/1</td></tr>
<tr><td>MLB</td><td>Def, Backer 24/1</td><td>Def, Six 25/1</td></tr>
<tr><td>LCB</td><td>Def, Corner 24/1</td><td>Def, Eight 25/1</td></tr>
</table>
<table><tr><th>Pos</th><th>Player</th><th>Player</th></tr>
<tr><td>WR</td><td>Squad, Receiver 25/7</td><td>Squad, Two 25/7</td></tr>
<tr><td>RB</td><td>Squad, Runner 25/7</td><td>Squad, Four 25/7</td></tr>
<tr><td>ED</td><td>Squad, Edge 25/7</td><td>Squad, Six 25/7</td></tr>
<tr><td>CB</td><td>Squad, Corner 25/7</td><td>Squad, Eight 25/7</td></tr>
<tr><td>S</td><td>Squad, Safety 25/7</td><td>Squad, Ten 25/7</td></tr>
</table>
</body></html>
--chart--
"""


class OurladsTableSelectionTests(unittest.TestCase):
    """Regression cover for the 2026-09-01 wrong-table bug.

    A saved page carries five same-shaped tables (offense / defense / special
    teams / practice squad / IR). Table choice used to be by SIZE, so the
    longest unit won - which is routinely the defense or the practice squad,
    not the offense. 24 of 32 real 2026 pages failed this way, reporting
    "no offense rows" or "no Pos / Player columns" with the offense table
    sitting unread right there. Both failure modes are pinned here because
    both were silent: the page parsed "successfully" into the wrong unit.
    """

    def setUp(self):
        self.frame, self.report = odc.parse_ourlads_depth_chart(
            _MULTI_UNIT_PAGE, "SEA.mhtml")

    def test_offense_table_is_chosen_over_a_longer_defense_table(self):
        self.assertEqual(self.report["error"], "")
        self.assertEqual(self.report["team"], "SEA")
        # Defense table has more rows than offense - size must not decide.
        players = set(self.frame["player"])
        self.assertIn("Starter Real", players)
        self.assertFalse([p for p in players if p.startswith("End Def")],
                         "defensive rows must never reach the snapshot")

    def test_practice_squad_table_is_excluded(self):
        # The PS table is the LONGEST here and contains real offense labels
        # (WR/RB), so a naive "most rows" or "any offense label" rule takes
        # it. Its rank-1 entries would then feed the role-floor logic as
        # starters.
        players = set(self.frame["player"])
        self.assertFalse([p for p in players if p.startswith("Receiver Squad")
                          or p.startswith("Runner Squad")],
                         "practice-squad rows must never reach the snapshot")
        self.assertEqual(len(self.frame), 4)  # LWR x2 + QB + RB

    def test_unnumbered_player_headers_still_parse(self):
        """A large minority of pages label columns "Player" with no number;
        the old number-only regex dropped those pages entirely."""
        page = _MULTI_UNIT_PAGE.replace(b"<th>Player 1</th><th>Player 2</th>",
                                        b"<th>Player</th><th>Player</th>", 1)
        frame, report = odc.parse_ourlads_depth_chart(page, "SEA.mhtml")
        self.assertEqual(report["error"], "")
        self.assertEqual(len(frame), 4)
        lwr = frame[frame["position_label"].eq("LWR")].sort_values("depth_rank")
        # Unnumbered columns take their left-to-right order as the rank.
        self.assertEqual(lwr["depth_rank"].tolist(), [1, 2])
        self.assertTrue(lwr.iloc[0]["is_listed_starter"])
        self.assertFalse(lwr.iloc[1]["is_listed_starter"])


class OurladsInboxAndArchiveTests(unittest.TestCase):
    def test_from_dir_imports_pages_and_archives_the_previous_snapshot(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inbox = tmp / "inbox"
            inbox.mkdir()
            (inbox / "2026 Miami Dolphins Depth Chart _ Ourlads.com.mhtml").write_bytes(_chart())
            csv_path = tmp / "ourlads_depth_charts.csv"
            csv_path.write_text("year,team\n2025,MIA\n")  # a "previous" snapshot to archive
            old_archive = odc.OURLADS_ARCHIVE_DIR
            odc.OURLADS_ARCHIVE_DIR = tmp / "archive"
            try:
                snap, report = odc.save_ourlads_snapshot_from_dir(inbox, year=2026, path=csv_path)
            finally:
                odc.OURLADS_ARCHIVE_DIR = old_archive

            self.assertEqual(report.get("error", ""), "")
            self.assertEqual(report["team_count"], 1)
            self.assertEqual(report["source_files"],
                             ["2026 Miami Dolphins Depth Chart _ Ourlads.com.mhtml"])
            # The fresh 2026 rows land...
            written = pd.read_csv(csv_path)
            self.assertTrue((written[written["year"] == 2026]["team"] == "MIA").all())
            # ...and an UNRELATED SEASON is left alone. Import merges rather
            # than replacing the file (changed 2026-09-01) - it only
            # supersedes the (year, team) pairs the batch actually contains.
            self.assertIn(2025, set(written["year"]))
            # the previous csv + the raw page were archived
            arch = report["archive"]
            self.assertTrue(Path(arch["archived_csv"]).is_file())
            self.assertEqual(arch["archived_pages"], 1)
            self.assertEqual(pd.read_csv(arch["archived_csv"])["year"].iloc[0], 2025)

    def test_batched_imports_accumulate_instead_of_replacing(self):
        """The league is ~74MB of pages, so it often cannot be imported in one
        go. Importing a few teams at a time must ADD to the snapshot; the old
        behaviour overwrote the file and silently discarded every earlier
        batch, while reporting the teams merely absent from the current batch
        as "missing"."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ourlads_depth_charts.csv"
            snap_a, rep_a = odc.save_ourlads_snapshot(
                [("mia.mhtml", _chart())], 2026, path=csv_path, archive=False)
            self.assertEqual(rep_a["batch_teams"], ["MIA"])
            self.assertEqual(rep_a["carried_teams"], [])

            # A second batch for a DIFFERENT team must not evict the first.
            snap_b, rep_b = odc.save_ourlads_snapshot(
                [("hou.mhtml", _chart(team="HOU", name="Houston Texans"))], 2026, path=csv_path, archive=False)
            self.assertEqual(rep_b["batch_teams"], ["HOU"])
            self.assertEqual(rep_b["carried_teams"], ["MIA"])
            self.assertEqual(rep_b["team_count"], 2)
            self.assertNotIn("MIA", rep_b["missing_teams"])

            on_disk, problem = odc.load_ourlads_snapshot(2026, path=csv_path)
            self.assertIsNone(problem)
            self.assertEqual(sorted(on_disk["team"].unique().tolist()), ["HOU", "MIA"])

            # Re-importing a team REPLACES its rows rather than duplicating.
            before = len(on_disk)
            odc.save_ourlads_snapshot(
                [("mia2.mhtml", _chart())], 2026, path=csv_path, archive=False)
            again, _ = odc.load_ourlads_snapshot(2026, path=csv_path)
            self.assertEqual(len(again), before)
            self.assertEqual(sorted(again["team"].unique().tolist()), ["HOU", "MIA"])


    def test_from_dir_reports_a_clear_error_when_the_folder_is_empty(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            snap, report = odc.save_ourlads_snapshot_from_dir(Path(tmp), year=2026)
            self.assertIn("No .mhtml", report["error"])
            self.assertTrue(snap.empty)


if __name__ == "__main__":
    unittest.main()
