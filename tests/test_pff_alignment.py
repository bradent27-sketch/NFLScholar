"""Focused offline tests for the time-safe local PFF alignment archive.

These fixtures intentionally write tiny temporary PFF-like CSVs rather than
reading the licensed exports in ``pff_imports/``.  They pin the important
guardrails: strict as-of filtering, regular-season provenance, no arbitrary
season-total fallback, stable PFF IDs, and neutral behavior when data is
missing or ambiguous.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import pff_alignment as pa  # noqa: E402


def _summary_row(**overrides):
    row = {
        "player": "Sample Receiver",
        "player_id": 1001,
        "position": "WR",
        "team_name": "KC",
        "player_game_count": 1,
        "slot_rate": 30.0,
        "wide_rate": 65.0,
        "inline_rate": 5.0,
        "slot_snaps": 30,
        "wide_snaps": 65,
        "inline_snaps": 5,
        "routes": 40,
        "targets": 7,
        "receptions": 5,
        "yards": 70,
        "touchdowns": 1,
    }
    row.update(overrides)
    return row


def _concept_row(**overrides):
    row = {
        "player": "Sample Receiver",
        "player_id": 1001,
        "position": "WR",
        "team_name": "KC",
        "slot_routes": 18,
        "slot_targets": 4,
        "slot_receptions": 3,
        "slot_yards": 45,
        "slot_touchdowns": 1,
    }
    row.update(overrides)
    return row


def _scheme_row(**overrides):
    row = {
        "player": "Sample Receiver",
        "player_id": 1001,
        "position": "WR",
        "team_name": "KC",
        "player_game_count": 1,
        "man_routes": 10,
        "man_targets": 3,
        "man_receptions": 2,
        "man_yards": 30,
        "man_touchdowns": 0,
        "man_yprr": 3.0,
        "zone_routes": 20,
        "zone_targets": 5,
        "zone_receptions": 4,
        "zone_yards": 60,
        "zone_touchdowns": 1,
        "zone_yprr": 3.0,
    }
    row.update(overrides)
    return row


def _write_week(root: Path, year: int, week: int, summary_rows=None, concept_rows=None, scheme_rows=None):
    week_dir = root / str(year) / "weekly" / str(week)
    week_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows is not None:
        pd.DataFrame(summary_rows).to_csv(week_dir / "receiving_summary.csv", index=False)
    if concept_rows is not None:
        pd.DataFrame(concept_rows).to_csv(week_dir / "receiving_concept.csv", index=False)
    if scheme_rows is not None:
        pd.DataFrame(scheme_rows).to_csv(week_dir / "receiving_scheme.csv", index=False)


def _write_manifest(root: Path, year: int, rows):
    weekly_dir = root / str(year) / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(weekly_dir / "manifest.csv", index=False)


def _write_season(root: Path, year: int, summary_rows=None, concept_rows=None):
    year_dir = root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows or [_summary_row()]).to_csv(year_dir / "receiving_summary.csv", index=False)
    pd.DataFrame(concept_rows or [_concept_row()]).to_csv(year_dir / "receiving_concept.csv", index=False)


def test_weekly_archive_is_strictly_as_of_and_aggregates_player_role():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row(slot_rate=30.0)], [_concept_row(slot_targets=4)])
        _write_week(root, 2026, 2, [_summary_row(slot_rate=50.0)], [_concept_row(slot_targets=8)])
        _write_manifest(root, 2026, [
            {"week": 1, "regular_season": True, "schema_valid": True},
            {"week": 2, "regular_season": True, "schema_valid": True},
        ])

        week_two = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        assert week_two.available and len(week_two.profiles) == 1
        row = week_two.profiles.iloc[0]
        assert row["source_weeks"] == "1" and row["source_week_count"] == 1
        assert math.isclose(row["slot_alignment_rate"], 0.30)
        assert math.isclose(row["non_slot_alignment_rate"], 0.70)
        assert row["slot_targets"] == 4.0
        assert row["alignment_matchup_multiplier"] == 1.0
        assert row["identity_quality"] == "pff_player_id"
        assert row["source_confidence"] == "weekly_manifest_regular_season"
        week_two_archive = week_two.archives.set_index("week")
        assert bool(week_two_archive.loc[1, "eligible_as_of"])
        assert not bool(week_two_archive.loc[2, "eligible_as_of"])
        assert any("strictly week < as_of_week" in issue for issue in week_two.issues)

        week_three = pa.load_weekly_alignment_profiles(2026, as_of_week=3, pff_root=root)
        aggregate = week_three.profiles.iloc[0]
        assert aggregate["source_weeks"] == "1,2"
        assert math.isclose(aggregate["slot_alignment_rate"], 0.40)
        assert aggregate["slot_targets"] == 12.0


def test_weekly_archive_rejects_manifested_nonregular_and_incomplete_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()])
        # Week 2 looks like a report but lacks the concept companion needed for
        # future defensive slot/non-slot construction.
        _write_week(root, 2026, 2, [_summary_row()], None)
        _write_manifest(root, 2026, [
            {"week": 1, "regular_season": False, "schema_valid": True},
            {"week": 2, "regular_season": True, "schema_valid": True},
        ])
        result = pa.load_weekly_alignment_profiles(2026, as_of_week=3, pff_root=root)
        assert result.profiles.empty and not result.available
        archive = result.archives.set_index("week")
        assert not bool(archive.loc[1, "eligible_as_of"])
        assert not bool(archive.loc[2, "pair_complete"])
        assert any("non-regular-season" in issue for issue in result.issues)
        assert any("incomplete" in issue for issue in result.issues)


def test_weekly_manifest_is_optional_but_its_status_stays_explicitly_unverified():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row(position="TE", slot_rate=47.0)], [_concept_row(position="TE")])
        result = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        assert result.available
        row = result.profiles.iloc[0]
        assert row["source_regular_season"] is None
        assert row["source_confidence"] == "weekly_schema_valid_regular_season_unverified"
        assert row["alignment_semantics"] == "slot / non-slot (non-slot is not asserted inline)"


def test_weekly_schema_failure_is_visible_and_never_becomes_a_zero_profile():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [{
            "player": "Sample Receiver", "player_id": 1001, "position": "WR", "team_name": "KC",
            # Deliberately no slot_routes: this could be a different PFF report.
            "slot_targets": 4,
        }])
        result = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        assert result.profiles.empty
        archive = result.archives.iloc[0]
        assert archive["schema_valid"] is False
        assert "slot_routes" in archive["schema_issues"]
        neutral = pa.lookup_alignment_profile(result.profiles, player_id=1001)
        assert not neutral["alignment_available"]
        assert neutral["alignment_matchup_multiplier"] == 1.0


def test_pff_id_preserves_a_player_across_team_change_while_name_fallback_stays_safe():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row(player_id=77, team_name="SEA", slot_rate=20.0)], [_concept_row(player_id=77, team_name="SEA")])
        _write_week(root, 2026, 2, [_summary_row(player_id=77, team_name="KC", slot_rate=40.0)], [_concept_row(player_id=77, team_name="KC")])
        result = pa.load_weekly_alignment_profiles(2026, as_of_week=3, pff_root=root)
        assert len(result.profiles) == 1
        row = result.profiles.iloc[0]
        assert row["team"] == "KC" and row["source_weeks"] == "1,2"
        assert math.isclose(row["slot_alignment_rate"], 0.30)
        assert pa.lookup_alignment_profile(result.profiles, player_id=77)["player"] == "Sample Receiver"
        assert pa.lookup_alignment_profile(result.profiles, player="Sample Receiver", team="NO")["alignment_available"] is False


def test_season_prior_requires_reviewed_regular_season_and_historical_time_validity():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_season(root, 2025)
        blocked = pa.load_season_alignment_prior(2025, 2026, 1, pff_root=root)
        assert blocked.profiles.empty
        assert any("regular-season-only status is unverified" in issue for issue in blocked.issues)

        reviewed = pa.load_season_alignment_prior(
            2025,
            2026,
            1,
            pff_root=root,
            source_metadata={"regular_season_only": True, "source_confidence": "reviewed REG export"},
        )
        assert reviewed.available and reviewed.profiles.iloc[0]["source_kind"] == "season_prior"
        assert reviewed.profiles.iloc[0]["source_week_count"] == 0
        assert bool(reviewed.profiles.iloc[0]["source_regular_season"])

        historical_blocked = pa.load_season_alignment_prior(
            2025,
            2026,
            1,
            pff_root=root,
            source_metadata={"regular_season": True},
            historical_backtest=True,
        )
        assert historical_blocked.profiles.empty
        assert any("time_valid=True" in issue for issue in historical_blocked.issues)

        historical_reviewed = pa.load_season_alignment_prior(
            2025,
            2026,
            1,
            pff_root=root,
            source_metadata={"regular_season": True, "time_valid": True},
            historical_backtest=True,
        )
        assert historical_reviewed.available


def test_alignment_efficiency_splits_slot_from_non_slot_and_withholds_below_minimum_sample():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        # Sample Receiver: total 12 targets/9 rec/100 yds; concept says 6/5/60
        # were in the slot, so non-slot is the remaining 6/4/40 - not a second
        # data source, just the summary total minus the concept report's
        # slot-only total (both sides clear the 5-target minimum here). Thin
        # Sample: stays below the minimum-target floor on BOTH sides (3 slot
        # targets, 2 non-slot) in the same weekly file.
        _write_week(root, 2026, 1, [
            _summary_row(targets=12, receptions=9, yards=100, touchdowns=1),
            _summary_row(player="Thin Sample", player_id=2002, targets=5, receptions=3, yards=40, touchdowns=0),
        ], [
            _concept_row(slot_targets=6, slot_receptions=5, slot_yards=60, slot_touchdowns=1),
            _concept_row(player="Thin Sample", player_id=2002,
                        slot_targets=3, slot_receptions=2, slot_yards=25, slot_touchdowns=0),
        ])
        _write_manifest(root, 2026, [{"week": 1, "regular_season": True, "schema_valid": True}])

        result = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        row = pa.lookup_alignment_profile(result.profiles, player_id=1001)
        assert math.isclose(row["non_slot_targets"], 6.0)
        assert math.isclose(row["non_slot_receptions"], 4.0)
        assert math.isclose(row["non_slot_yards"], 40.0)
        assert math.isclose(row["slot_catch_rate"], 5.0 / 6.0)
        assert math.isclose(row["slot_yards_per_target"], 10.0)
        assert math.isclose(row["non_slot_catch_rate"], 4.0 / 6.0)
        assert math.isclose(row["non_slot_yards_per_target"], 40.0 / 6.0)

        # A row read back off an aggregated (non-empty) profiles DataFrame -
        # unlike neutral_alignment_profile()'s raw dict - has this column at
        # float dtype once ANY row in the pool has a real value, so a withheld
        # cell here is NaN, not the Python None a dict lookup would show.
        thin_row = pa.lookup_alignment_profile(result.profiles, player_id=2002)
        assert pd.isna(thin_row["slot_catch_rate"]) and pd.isna(thin_row["slot_yards_per_target"])
        assert pd.isna(thin_row["non_slot_catch_rate"]) and pd.isna(thin_row["non_slot_yards_per_target"])
        assert math.isclose(thin_row["non_slot_targets"], 2.0)


def test_save_weekly_alignment_export_writes_files_and_manifest_then_loads_back():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        summary_csv = io.StringIO(pd.DataFrame([_summary_row()]).to_csv(index=False))
        concept_csv = io.StringIO(pd.DataFrame([_concept_row()]).to_csv(index=False))

        ok, issues = pa.save_weekly_alignment_export(summary_csv, concept_csv, 2026, 1, pff_root=root)
        assert ok and not issues
        assert (root / "2026" / "weekly" / "1" / "receiving_summary.csv").is_file()
        assert (root / "2026" / "weekly" / "1" / "receiving_concept.csv").is_file()
        manifest = pd.read_csv(root / "2026" / "weekly" / "manifest.csv")
        assert manifest.loc[0, "week"] == 1
        assert bool(manifest.loc[0, "regular_season"]) and bool(manifest.loc[0, "schema_valid"])

        # Loads back through the normal read path exactly like a manually
        # placed file would - the upload path is not a special case downstream.
        result = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        assert result.available and len(result.profiles) == 1

        # Re-uploading the SAME week replaces its manifest row rather than
        # duplicating it.
        ok2, _issues2 = pa.save_weekly_alignment_export(
            io.StringIO(pd.DataFrame([_summary_row(slot_rate=55.0)]).to_csv(index=False)),
            io.StringIO(pd.DataFrame([_concept_row()]).to_csv(index=False)),
            2026, 1, pff_root=root,
        )
        assert ok2
        manifest_again = pd.read_csv(root / "2026" / "weekly" / "manifest.csv")
        assert len(manifest_again) == 1


def test_save_weekly_alignment_export_rejects_a_bad_schema_before_writing_anything():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        bad_summary = io.StringIO(pd.DataFrame([{"player": "X"}]).to_csv(index=False))  # no slot_rate, no position
        concept_csv = io.StringIO(pd.DataFrame([_concept_row()]).to_csv(index=False))
        ok, issues = pa.save_weekly_alignment_export(bad_summary, concept_csv, 2026, 1, pff_root=root)
        assert not ok and issues
        assert not (root / "2026" / "weekly" / "1").exists()


def test_offensive_weekly_archive_builds_schedule_mapped_defense_profiles_and_neutral_preview():
    """Only offensive weekly reports may create the experimental defense side."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        # KC's same WR role is slot-heavy against DEN in Week 1 and non-slot
        # heavy against BUF in Week 2.  The other time-valid KC game provides
        # the leave-one-out offensive-quality comparison for each defense.
        _write_week(root, 2026, 1, [
            _summary_row(team_name="KC", slot_rate=80.0, wide_rate=20.0, inline_rate=0.0,
                         slot_snaps=80, wide_snaps=20, inline_snaps=0,
                         targets=10, receptions=8, yards=100, touchdowns=1),
        ], [
            _concept_row(team_name="KC", slot_targets=8, slot_receptions=6, slot_yards=80, slot_touchdowns=0),
        ])
        _write_week(root, 2026, 2, [
            _summary_row(team_name="KC", slot_rate=20.0, wide_rate=80.0, inline_rate=0.0,
                         slot_snaps=20, wide_snaps=80, inline_snaps=0,
                         targets=10, receptions=8, yards=100, touchdowns=1),
        ], [
            _concept_row(team_name="KC", slot_targets=2, slot_receptions=2, slot_yards=20, slot_touchdowns=0),
        ])
        _write_manifest(root, 2026, [
            {"week": 1, "regular_season": True, "schema_valid": True},
            {"week": 2, "regular_season": True, "schema_valid": True},
        ])
        schedule = pd.DataFrame([
            {"week": 1, "home_team": "KC", "away_team": "DEN", "game_type": "REG"},
            {"week": 2, "home_team": "KC", "away_team": "BUF", "game_type": "REG"},
        ])

        result = pa.load_weekly_alignment_defense_profiles(2026, 3, schedule, pff_root=root)
        assert result.available
        assert result.metadata["scoring_active"] is False
        assert result.metadata["defender_slot_coverage_used"] is False
        assert set(result.team_games["alignment"]) == {"slot", "non_slot"}
        week_one = result.team_games[
            (result.team_games["source_week"] == 1)
            & (result.team_games["offense_team"] == "KC")
            & (result.team_games["position"] == "WR")
            & (result.team_games["stat"] == "targets")
        ].set_index("alignment")
        assert week_one.loc["slot", "defense_team"] == "DEN"
        assert week_one.loc["slot", "observed_value"] == 8.0
        assert week_one.loc["non_slot", "observed_value"] == 2.0
        assert "receiving_summary.csv" in week_one.loc["slot", "source_paths"]
        assert "receiving_concept.csv" in week_one.loc["slot", "source_paths"]

        den_slot = pa.lookup_alignment_defense_profile(
            result.profiles, defense_team="DEN", position="WR", alignment="slot", stat="targets"
        )
        assert den_slot["profile_available"]
        assert den_slot["candidate_available"]
        assert math.isclose(den_slot["raw_allowed_ratio"], 4.0)
        assert 0.0 < den_slot["shrinkage_weight"] < 1.0
        assert 1.0 < den_slot["shrunk_allowed_ratio"] < den_slot["raw_allowed_ratio"]
        assert den_slot["baseline_source"] == "offense_leave_one_out_time_valid"
        assert den_slot["position_normal_slot_rate"] == 0.5

        preview = pa.alignment_defense_residual_multiplier(
            result.profiles, defense_team="DEN", position="WR", player_slot_rate=0.80, stat="receiving_yards"
        )
        assert preview["candidate_available"]
        assert preview["candidate_multiplier"] > 1.0
        assert preview["multiplier"] == 1.0
        assert not preview["applied"]
        td_preview = pa.alignment_defense_residual_multiplier(
            result.profiles, defense_team="DEN", position="WR", player_slot_rate=0.80, stat="receiving_tds"
        )
        assert td_preview["candidate_multiplier"] == 1.0
        assert td_preview["multiplier"] == 1.0
        assert not td_preview["candidate_available"]

        # A Week 2 projection can only see Week 1.  With no other KC game
        # for comparison, the profile remains visible but safely neutral.
        week_two = pa.load_weekly_alignment_defense_profiles(2026, 2, schedule, pff_root=root)
        assert not week_two.available
        early = pa.lookup_alignment_defense_profile(
            week_two.profiles, defense_team="DEN", position="WR", alignment="slot", stat="targets"
        )
        assert not early["profile_available"]
        assert early["shrunk_allowed_ratio"] == 1.0


def test_alignment_defense_missing_schedule_or_source_is_neutral_not_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()])
        result = pa.load_weekly_alignment_defense_profiles(2026, 2, None, pff_root=root)
        assert not result.available
        assert result.team_games.empty and result.profiles.empty
        assert any("schedule_df" in issue for issue in result.issues)
        neutral = pa.lookup_alignment_defense_profile(
            result.profiles, defense_team="HST", position="WR", alignment="slot", stat="targets"
        )
        assert neutral["defense_team"] == "HOU"
        assert not neutral["profile_available"]
        assert neutral["shrunk_allowed_ratio"] == 1.0
        preview = pa.alignment_defense_residual_multiplier(
            result.profiles, defense_team="HOU", position="WR", player_slot_rate=0.5, stat="yards"
        )
        assert preview["multiplier"] == 1.0
        assert preview["candidate_multiplier"] == 1.0
        assert not preview["candidate_available"]


def test_weekly_scheme_archive_is_strictly_as_of_and_aggregates_route_share():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        # Week 1: 10 man routes / 20 zone routes -> man share 1/3.
        # Week 2: 20 man routes / 10 zone routes -> man share 2/3.
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()],
                    [_scheme_row(man_routes=10, zone_routes=20, man_targets=3, zone_targets=5)])
        _write_week(root, 2026, 2, [_summary_row()], [_concept_row()],
                    [_scheme_row(man_routes=20, zone_routes=10, man_targets=6, zone_targets=2)])
        _write_manifest(root, 2026, [
            {"week": 1, "regular_season": True, "schema_valid": True},
            {"week": 2, "regular_season": True, "schema_valid": True},
        ])

        week_two = pa.load_weekly_scheme_profiles(2026, as_of_week=2, pff_root=root)
        assert week_two.available and len(week_two.profiles) == 1
        row = week_two.profiles.iloc[0]
        assert row["source_weeks"] == "1" and row["source_week_count"] == 1
        assert math.isclose(row["man_route_share"], 1.0 / 3.0)
        assert math.isclose(row["zone_route_share"], 2.0 / 3.0)
        assert row["man_routes"] == 10.0 and row["zone_routes"] == 20.0
        assert row["identity_quality"] == "pff_player_id"

        # Equal route volume both weeks (30 each) -> plain average of the two
        # weekly shares, 1/3 and 2/3, lands exactly on 0.5.
        week_three = pa.load_weekly_scheme_profiles(2026, as_of_week=3, pff_root=root)
        aggregate = week_three.profiles.iloc[0]
        assert aggregate["source_weeks"] == "1,2"
        assert math.isclose(aggregate["man_route_share"], 0.5)
        assert math.isclose(aggregate["zone_route_share"], 0.5)
        assert aggregate["man_routes"] == 30.0 and aggregate["zone_routes"] == 30.0


def test_scheme_route_share_ignores_pffs_own_route_rate_columns_not_a_slot_style_split():
    # Regression pin: PFF's man_route_rate/zone_route_rate are each "routes
    # run DIVIDED BY that coverage's own pass_plays" (route participation
    # conditional on the defense's call), not a share of this player's total
    # routes - real exports do not sum to 1.0 (e.g. 100.0 and 96.4 for the
    # same player-week). A loader that trusted those columns directly would
    # produce a nonsensical "tendency" that doesn't sum to 1 across players.
    # The real man_route_share must come from man_routes/zone_routes instead.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()], [
            _scheme_row(man_routes=10, zone_routes=27, man_route_rate=100.0, zone_route_rate=96.4)
        ])
        _write_manifest(root, 2026, [{"week": 1, "regular_season": True, "schema_valid": True}])

        result = pa.load_weekly_scheme_profiles(2026, as_of_week=2, pff_root=root)
        row = result.profiles.iloc[0]
        assert math.isclose(row["man_route_share"], 10.0 / 37.0)
        assert math.isclose(row["zone_route_share"], 27.0 / 37.0)
        assert math.isclose(row["man_route_share"] + row["zone_route_share"], 1.0)


def test_scheme_efficiency_withholds_below_minimum_sample():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()], [
            _scheme_row(man_targets=6, man_receptions=5, man_yards=60,
                        zone_targets=2, zone_receptions=1, zone_yards=15),
            _scheme_row(player="Thin Sample", player_id=2002,
                        man_targets=3, man_receptions=2, man_yards=25,
                        zone_targets=4, zone_receptions=3, zone_yards=40),
        ])
        _write_manifest(root, 2026, [{"week": 1, "regular_season": True, "schema_valid": True}])

        result = pa.load_weekly_scheme_profiles(2026, as_of_week=2, pff_root=root)
        row = result.profiles[result.profiles["player_id"] == "1001"].iloc[0]
        assert math.isclose(row["man_catch_rate"], 5.0 / 6.0)
        assert math.isclose(row["man_yards_per_target"], 10.0)
        assert pd.isna(row["zone_catch_rate"]) and pd.isna(row["zone_yards_per_target"])

        thin_row = result.profiles[result.profiles["player_id"] == "2002"].iloc[0]
        assert pd.isna(thin_row["man_catch_rate"]) and pd.isna(thin_row["man_yards_per_target"])
        assert pd.isna(thin_row["zone_catch_rate"]) and pd.isna(thin_row["zone_yards_per_target"])


def test_scheme_profile_skips_a_week_missing_receiving_scheme_but_alignment_stays_intact():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        # Week 1 has the summary/concept pair but no receiving_scheme.csv -
        # backward compatible: the existing slot/wide/inline archive must be
        # unaffected, and the scheme loader must skip the week, not error.
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()])
        _write_manifest(root, 2026, [{"week": 1, "regular_season": True, "schema_valid": True}])

        alignment = pa.load_weekly_alignment_profiles(2026, as_of_week=2, pff_root=root)
        assert alignment.available and len(alignment.profiles) == 1

        scheme = pa.load_weekly_scheme_profiles(2026, as_of_week=2, pff_root=root)
        assert not scheme.available
        assert scheme.profiles.empty
        assert any("no receiving_scheme.csv" in issue for issue in scheme.issues)


def test_offensive_weekly_archive_builds_schedule_mapped_scheme_defense_profiles_and_neutral_preview():
    """Man/zone twin of test_offensive_weekly_archive_builds_schedule_mapped_defense_profiles_and_neutral_preview."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        # KC's same WR is man-heavy against DEN in Week 1 and zone-heavy
        # against BUF in Week 2 - the other time-valid KC game provides the
        # leave-one-out offensive-quality comparison for each defense, same
        # shape as the alignment-defense test this mirrors.
        _write_week(root, 2026, 1, [_summary_row(team_name="KC")], [_concept_row(team_name="KC")], [
            _scheme_row(team_name="KC", man_targets=8, man_receptions=6, man_yards=80, man_touchdowns=0,
                       zone_targets=2, zone_receptions=2, zone_yards=20, zone_touchdowns=0,
                       man_routes=40, zone_routes=10),
        ])
        _write_week(root, 2026, 2, [_summary_row(team_name="KC")], [_concept_row(team_name="KC")], [
            _scheme_row(team_name="KC", man_targets=2, man_receptions=2, man_yards=20, man_touchdowns=0,
                       zone_targets=8, zone_receptions=6, zone_yards=80, zone_touchdowns=0,
                       man_routes=10, zone_routes=40),
        ])
        _write_manifest(root, 2026, [
            {"week": 1, "regular_season": True, "schema_valid": True},
            {"week": 2, "regular_season": True, "schema_valid": True},
        ])
        schedule = pd.DataFrame([
            {"week": 1, "home_team": "KC", "away_team": "DEN", "game_type": "REG"},
            {"week": 2, "home_team": "KC", "away_team": "BUF", "game_type": "REG"},
        ])

        result = pa.load_weekly_scheme_defense_profiles(2026, 3, schedule, pff_root=root)
        assert result.available
        assert result.metadata["scoring_active"] is False
        assert set(result.team_games["scheme"]) == {"man", "zone"}
        week_one = result.team_games[
            (result.team_games["source_week"] == 1)
            & (result.team_games["offense_team"] == "KC")
            & (result.team_games["position"] == "WR")
            & (result.team_games["stat"] == "targets")
        ].set_index("scheme")
        assert week_one.loc["man", "defense_team"] == "DEN"
        assert week_one.loc["man", "observed_value"] == 8.0
        assert week_one.loc["zone", "observed_value"] == 2.0

        den_man = pa.lookup_scheme_defense_profile(
            result.profiles, defense_team="DEN", position="WR", scheme="man", stat="targets"
        )
        assert den_man["profile_available"]
        assert den_man["candidate_available"]
        assert math.isclose(den_man["raw_allowed_ratio"], 4.0)
        assert 0.0 < den_man["shrinkage_weight"] < 1.0
        assert 1.0 < den_man["shrunk_allowed_ratio"] < den_man["raw_allowed_ratio"]
        assert den_man["baseline_source"] == "offense_leave_one_out_time_valid"
        assert den_man["position_normal_man_rate"] == 0.5

        preview = pa.scheme_defense_residual_multiplier(
            result.profiles, defense_team="DEN", position="WR", player_man_rate=0.80, stat="yards"
        )
        assert preview["candidate_available"]
        assert preview["multiplier"] == 1.0
        assert preview["scoring_active"] is False


def test_scheme_defense_missing_schedule_or_source_is_neutral_not_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pff_imports"
        _write_week(root, 2026, 1, [_summary_row()], [_concept_row()], [_scheme_row()])
        result = pa.load_weekly_scheme_defense_profiles(2026, 2, None, pff_root=root)
        assert not result.available
        assert result.team_games.empty and result.profiles.empty
        assert any("schedule_df" in issue for issue in result.issues)
        neutral = pa.lookup_scheme_defense_profile(
            result.profiles, defense_team="HST", position="WR", scheme="man", stat="targets"
        )
        assert neutral["defense_team"] == "HOU"
        assert not neutral["profile_available"]
        assert neutral["shrunk_allowed_ratio"] == 1.0
        preview = pa.scheme_defense_residual_multiplier(
            result.profiles, defense_team="HOU", position="WR", player_man_rate=0.5, stat="yards"
        )
        assert preview["multiplier"] == 1.0
        assert preview["candidate_multiplier"] == 1.0
        assert not preview["candidate_available"]


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
