"""Focused contracts for the Week-1 availability resolver.

Ourlads styling is useful source context, but it is deliberately not an
availability feed.  These checks make sure only a current/manual profile can
zero a role and that a suffix spelling does not create a second player.
"""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.availability_overrides import resolve_target_week_availability  # noqa: E402


class AvailabilityOverrideTests(unittest.TestCase):
    def setUp(self):
        self.roster = pd.DataFrame([
            {'player_display_name': 'James Cook', 'team': 'BUF', 'position': 'RB',
             'gsis_id': '00-0037248'},
            {'player_display_name': 'Malik Nabers', 'team': 'NYG', 'position': 'WR',
             'gsis_id': '00-0039406'},
        ])

    def test_unique_suffix_identity_updates_existing_roster_player(self):
        profiles = {'James Cook III': {'status': 'out', 'plays_probability': 0.0}}
        resolved, warnings = resolve_target_week_availability(
            profiles, pd.DataFrame(), self.roster,
            'player_display_name', 'team',
        )
        self.assertFalse(warnings)
        self.assertEqual(set(resolved), {'James Cook'})
        self.assertEqual(resolved['James Cook']['status'], 'out')
        self.assertEqual(resolved['James Cook']['match_method'], 'unique suffix-stripped full name')

    def test_chart_styling_cannot_zero_an_active_player(self):
        # The resolver is intentionally not handed any Ourlads CSS/status
        # field.  An empty current availability feed stays neutral.
        resolved, _ = resolve_target_week_availability(
            {}, pd.DataFrame(), self.roster,
            'player_display_name', 'team',
        )
        self.assertEqual(resolved, {})

    def test_manual_current_override_wins_and_is_auditable(self):
        manual = pd.DataFrame([
            {'year': 2026, 'week': 1, 'team': 'NYG', 'player': 'Malik Nabers',
             'status': 'out', 'plays_probability': 0.0, 'workload_if_active': 0.0,
             'note': 'confirmed test fixture'},
        ])
        resolved, warnings = resolve_target_week_availability(
            {'Malik Nabers': {'status': 'questionable', 'plays_probability': 0.75}},
            manual, self.roster, 'player_display_name', 'team',
        )
        self.assertFalse(warnings)
        self.assertEqual(resolved['Malik Nabers']['status'], 'out')
        self.assertEqual(resolved['Malik Nabers']['source'], 'manual target-week availability override')
        self.assertEqual(resolved['Malik Nabers']['note'], 'confirmed test fixture')


if __name__ == '__main__':
    unittest.main()
