"""Regression tests for the conservative Ourlads/current-roster identity layer."""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.ourlads_depth_charts as odc  # noqa: E402
from data.player_aliases import stable_roster_identity_keys  # noqa: E402


def _chart(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Make the minimal literal-source shape used by the resolver."""
    output = []
    for index, row in enumerate(rows, start=1):
        player = str(row["player"])
        position = str(row.get("position", "RB"))
        depth_rank = int(row.get("depth_rank", 1))
        output.append({
            "team": row.get("team", "BUF"),
            "position": position,
            "position_label": row.get("position_label", position),
            "depth_rank": depth_rank,
            "source_row": int(row.get("source_row", index)),
            "source_slot": int(row.get("source_slot", depth_rank)),
            "position_occurrence": int(row.get("position_occurrence", 0)),
            "is_listed_starter": bool(row.get("is_listed_starter", depth_rank == 1)),
            "is_inactive": bool(row.get("is_inactive", False)),
            "status_class": row.get("status_class", "lc_red" if row.get("is_inactive", False) else ""),
            "source_player_id": row.get("source_player_id", ""),
            "player": player,
            "player_key": odc.clean_name_exact(pd.Series([player])).iloc[0],
            "source_gsis_id": row.get("source_gsis_id", ""),
            "source_pff_id": row.get("source_pff_id", ""),
        })
    return pd.DataFrame(output)


class OurladsIdentityResolutionTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([
            {"name": "James Cook", "team": "BUF", "position": "RB", "gsis_id": "00-0037248", "pff_id": "9991"},
            {"name": "Kenneth Walker III", "team": "KC", "position": "RB", "gsis_id": "00-0038134", "pff_id": "9992"},
            {"name": "Travis Etienne", "team": "NO", "position": "RB", "gsis_id": "00-0036973", "pff_id": "9993"},
        ])

    def test_suffix_variants_resolve_to_existing_roster_identities_without_appending(self):
        source = _chart([
            {"player": "James Cook III", "team": "BUF"},
            {"player": "Kenneth Walker", "team": "KC"},
            {"player": "Travis Etienne Jr.", "team": "NO"},
        ])
        matches, audit, warnings = odc.resolve_ourlads_roster_identities(source, self.roster, "name", "team")

        self.assertFalse(warnings)
        self.assertEqual(matches["matched_player"].tolist(), [
            "James Cook", "Kenneth Walker III", "Travis Etienne",
        ])
        self.assertEqual(matches["matched_gsis_id"].tolist(), [
            "00-0037248", "00-0038134", "00-0036973",
        ])
        self.assertEqual(set(matches["match_method"]), {"suffix-stripped name"})
        self.assertEqual(set(audit["match_status"]), {"matched"})

        overlaid, changes, overlay_warnings = odc.apply_ourlads_starter_roster_overlay(
            source, self.roster, "name", "team")
        self.assertFalse(overlay_warnings)
        self.assertEqual(len(overlaid), len(self.roster))
        self.assertEqual(set(overlaid["gsis_id"]), set(self.roster["gsis_id"]))
        self.assertFalse(any(change["action"] == "added verified missing starter" for change in changes))

    def test_gsis_id_beats_stale_team_or_display_name(self):
        source = _chart([{
            "player": "Cook, James III", "team": "BUF", "source_gsis_id": "00-0037248",
        }])
        stale_roster = self.roster.copy()
        stale_roster.loc[0, "team"] = "OLD"
        matches, audit, _ = odc.resolve_ourlads_roster_identities(source, stale_roster, "name", "team")
        self.assertEqual(matches.iloc[0]["matched_player"], "James Cook")
        self.assertEqual(matches.iloc[0]["match_method"], "GSIS ID")
        self.assertEqual(audit.iloc[0]["match_confidence"], "high")

        overlaid, changes, warnings = odc.apply_ourlads_starter_roster_overlay(
            source, stale_roster, "name", "team")
        self.assertFalse(warnings)
        self.assertEqual(overlaid.iloc[0]["team"], "BUF")
        self.assertEqual(changes[0]["match_method"], "GSIS ID")

    def test_ambiguous_suffix_match_stays_unresolved_and_auditable(self):
        source = _chart([{"player": "Byron Murphy III", "team": "SEA", "position": "RB"}])
        roster = pd.DataFrame([
            {"name": "Byron Murphy", "team": "SEA", "position": "RB", "gsis_id": "one"},
            {"name": "Byron Murphy II", "team": "SEA", "position": "RB", "gsis_id": "two"},
        ])
        matches, audit, warnings = odc.resolve_ourlads_roster_identities(source, roster, "name", "team")
        self.assertTrue(matches.empty)
        self.assertEqual(audit.iloc[0]["match_status"], "ambiguous")
        self.assertIn("suffix-stripped", audit.iloc[0]["match_warning"])
        self.assertTrue(any("identity is ambiguous" in warning for warning in warnings))

    def test_red_chart_status_preserves_rb1_role_until_current_availability_confirms_out(self):
        source = _chart([{
            "player": "Jeremiyah Love", "team": "ARI", "position": "RB", "depth_rank": 1,
            "is_inactive": True,
        }])
        roster = pd.DataFrame([{
            "name": "Jeremiyah Love", "team": "ARI", "position": "RB", "gsis_id": "00-0041027",
        }])
        signal = odc.build_ourlads_projection_signal(source, roster, "name", "team")
        self.assertEqual(signal["matches"].iloc[0]["source_rank"], 1)
        self.assertIn("Jeremiyah Love", set(signal["skill_roles"]["matched_player"]))
        self.assertIn("current availability must confirm", signal["matches"].iloc[0]["source_status_warning"])
        self.assertEqual(signal["source_roles"].iloc[0]["match_status"], "matched")

    def test_red_qb_remains_the_chart_qb1_signal_until_current_availability_confirms_out(self):
        source = _chart([{
            "player": "Michael Penix Jr.", "team": "ATL", "position": "QB", "depth_rank": 1,
            "is_inactive": True,
        }])
        roster = pd.DataFrame([
            {"name": "Michael Penix Jr.", "team": "ATL", "position": "QB", "gsis_id": "00-0039950"},
            {"name": "Kirk Cousins", "team": "ATL", "position": "QB", "gsis_id": "00-0029604"},
        ])
        signal = odc.build_ourlads_projection_signal(source, roster, "name", "team")
        self.assertEqual(signal["qb_starters"].iloc[0]["matched_player"], "Michael Penix Jr.")
        self.assertTrue(signal["qb_starters"].iloc[0]["source_is_inactive"])

    def test_stable_roster_identity_key_deduplicates_same_gsis_not_display_name(self):
        rows = pd.DataFrame([
            {"name": "James Cook", "gsis_id": "00-0037248"},
            {"name": "James Cook III", "gsis_id": "00-0037248"},
            {"name": "James Cook III", "gsis_id": "00-0099999"},
        ])
        keys = stable_roster_identity_keys(rows, "name")
        self.assertEqual(keys.iloc[0], keys.iloc[1])
        self.assertNotEqual(keys.iloc[1], keys.iloc[2])

    def test_gsis_and_historical_player_id_share_one_identity_namespace(self):
        # nflverse weekly files use player_id for the same GSIS value that
        # modern roster files expose as gsis_id.  Cross-season role evidence
        # must survive that harmless schema difference.
        current = pd.DataFrame([
            {'name': 'James Cook', 'gsis_id': '00-0037248'},
        ])
        historic = pd.DataFrame([
            {'name': 'James Cook III', 'player_id': '00-0037248'},
        ])
        self.assertEqual(
            stable_roster_identity_keys(current, 'name').iloc[0],
            stable_roster_identity_keys(historic, 'name').iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
