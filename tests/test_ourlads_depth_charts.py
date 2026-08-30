"""Focused offline tests for Ourlads fullback and literal-rank evidence."""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.ourlads_depth_charts as odc  # noqa: E402


def _chart() -> bytes:
    return b'''From: <Saved by Blink>
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
            # live snapshot was overwritten with the fresh 2026 rows
            self.assertTrue((pd.read_csv(csv_path)["year"] == 2026).all())
            # the previous csv + the raw page were archived
            arch = report["archive"]
            self.assertTrue(Path(arch["archived_csv"]).is_file())
            self.assertEqual(arch["archived_pages"], 1)
            self.assertEqual(pd.read_csv(arch["archived_csv"])["year"].iloc[0], 2025)

    def test_from_dir_reports_a_clear_error_when_the_folder_is_empty(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            snap, report = odc.save_ourlads_snapshot_from_dir(Path(tmp), year=2026)
            self.assertIn("No .mhtml", report["error"])
            self.assertTrue(snap.empty)


if __name__ == "__main__":
    unittest.main()
