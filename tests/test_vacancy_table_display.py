"""Deep Dive "How vacated volume is redistributed" table - display trims.

Covers the pure decision helper `_summarize_vacancy_entry` (no Streamlit) and
the `team_rank` metadata the ledger producers now attach for it.  Requested
2026-08-30: a small-source OUT player is not worth a full distribution, and
recipients below the trusted tier get rolled into one line.  Tightened
2026-08-31: the "small source" bar is >1 vacated target / carry per game (was
1.5 / 3.0), and it now collapses even when the open player is a tiny
recipient - the one-line form is the informative caption, not a stub.
"""

import numpy as np
import pandas as pd

import data.weekly_projections as wp
from data.pass_capacity_allocator import (
    PASS_CAPACITY_TRUSTED_TIER, PASS_CAPACITY_TRUSTED_TIER_RB)
from ui.tabs.rankings import (
    _summarize_vacancy_entry, _vacancy_trusted_tier_cutoff)


def _entry(**over):
    base = dict(team='KC', volume='targets', source_player='Out WR',
                vacated=6.0, allocated=5.0, unallocated=1.0,
                recipients=[], reason='Role-compatible as-of projected recipients.')
    base.update(over)
    return base


def _rec(player, allocated, rank):
    return {'player': player, 'allocated': allocated, 'team_rank': rank}


# --- _vacancy_trusted_tier_cutoff -----------------------------------------

def test_tier_cutoff_is_position_group_aware():
    assert _vacancy_trusted_tier_cutoff('rushing_attempts', None) == PASS_CAPACITY_TRUSTED_TIER_RB
    assert _vacancy_trusted_tier_cutoff('targets', 'WR') == PASS_CAPACITY_TRUSTED_TIER
    assert _vacancy_trusted_tier_cutoff('targets', 'RB') == PASS_CAPACITY_TRUSTED_TIER_RB
    assert _vacancy_trusted_tier_cutoff('passing_attempts', None) is None


# --- negligible source ---------------------------------------------------

def test_sub_threshold_target_source_collapses_to_one_line():
    small = _entry(vacated=0.6, allocated=0.6, unallocated=0.0,
                   source_player='John Michael Gyllenborg',
                   recipients=[_rec('WR One', 0.6, 1)],
                   reason='Allocator-weighted core-RB injury redistribution.')
    # Not a recipient -> one-line, and the line is the informative caption
    # (source, vacated, redistributed, unfilled, reason), not a stub.
    summ = _summarize_vacancy_entry(small, this_player='Someone Else')
    assert summ['kind'] == 'negligible'
    assert summ['text'] == ('John Michael Gyllenborg out — 0.6 targets vacated; '
                            '0.6 redistributed to active teammates, 0.0 left unfilled. '
                            'Allocator-weighted core-RB injury redistribution.')

    # New 2026-08-31: the open player being one of the (tiny) recipients no
    # longer forces the full grid - a 0.6-target vacancy is still one line.
    summ = _summarize_vacancy_entry(small, this_player='WR One')
    assert summ['kind'] == 'negligible'

    # Just over the 1-per-game bar -> full distribution.
    summ = _summarize_vacancy_entry(_entry(vacated=1.4, source_player='Real WR',
                                           recipients=[_rec('WR One', 1.2, 1)]),
                                    this_player='Someone Else')
    assert summ['kind'] == 'full'


def test_carry_source_uses_a_one_per_game_bar():
    small = _summarize_vacancy_entry(
        _entry(volume='rushing_attempts', vacated=0.7,
               recipients=[_rec('RB One', 0.7, 1)]), this_player='x')
    assert small['kind'] == 'negligible'
    big = _summarize_vacancy_entry(
        _entry(volume='rushing_attempts', vacated=1.5,
               recipients=[_rec('RB One', 1.4, 1)]), this_player='x')
    assert big['kind'] == 'full'


def test_missing_qb_is_never_negligible():
    summ = _summarize_vacancy_entry(
        _entry(volume='passing_attempts', vacated=0.6,
               recipients=[_rec('Backup QB', 0.6, 1)]), this_player='x')
    assert summ['kind'] == 'full'
    assert len(summ['recipients']) == 1


# --- trusted-tier recipient trim ---------------------------------------

def test_receiver_recipients_below_tier_eight_roll_into_one_line():
    recips = [_rec('WR1', 3.0, 1), _rec('WR2', 1.5, 5),
              _rec('WR9', 0.3, 9), _rec('WR12', 0.2, 12)]
    summ = _summarize_vacancy_entry(_entry(recipients=recips), this_player='WR1')
    kept = {r['Fills in for them'] for r in summ['recipients']}
    assert kept == {'WR1', 'WR2'}
    assert summ['minor'] is not None
    assert '2 more fill-ins below the trusted tier absorbed +0.50' in summ['minor']


def test_carry_recipients_below_rb_tier_two_roll_up():
    recips = [_rec('RB1', 6.0, 1), _rec('RB2', 2.0, 2), _rec('RB3', 0.4, 3)]
    summ = _summarize_vacancy_entry(
        _entry(volume='rushing_attempts', vacated=10.0,
               functional_source_role='RB', recipients=recips),
        this_player='RB1')
    kept = {r['Fills in for them'] for r in summ['recipients']}
    assert kept == {'RB1', 'RB2'}
    assert '1 more fill-in below the trusted tier' in summ['minor']


def test_this_player_is_kept_even_below_the_tier():
    recips = [_rec('WR1', 3.0, 1), _rec('Deep Guy', 0.3, 15)]
    summ = _summarize_vacancy_entry(_entry(recipients=recips), this_player='Deep Guy')
    kept = {r['Fills in for them'] for r in summ['recipients']}
    assert kept == {'WR1', 'Deep Guy'}
    assert summ['minor'] is None


def test_recipients_without_rank_metadata_are_all_kept():
    recips = [{'player': 'WR1', 'allocated': 3.0}, {'player': 'WR2', 'allocated': 1.0}]
    summ = _summarize_vacancy_entry(_entry(recipients=recips), this_player='WR1')
    assert len(summ['recipients']) == 2 and summ['minor'] is None


def test_skipped_pass_is_reported_verbatim():
    summ = _summarize_vacancy_entry(
        _entry(vacated=0.0, allocated=0.0, unallocated=0.0, recipients=[],
               reason='No projected, active role-compatible replacement.'),
        this_player='x')
    assert summ['kind'] == 'skipped'
    assert 'No projected, active role-compatible replacement.' in summ['text']


# --- ledger producers attach team_rank -------------------------------

def test_redistribute_v2_vacated_usage_tags_recipient_team_rank():
    frame = pd.DataFrame({
        'Player': ['Out WR', 'WR Big', 'WR Small'],
        'Pos': ['WR', 'WR', 'WR'],
        'Team': ['KC', 'KC', 'KC'],
        'Expected Snap Share': [0.9, 0.8, 0.5],
        'targets': [0.0, 8.0, 2.0],
        'receptions': [0.0, 5.0, 1.0],
        'receiving_yards': [0.0, 60.0, 12.0],
    })
    frame['_full_targets'] = [9.0, 8.0, 2.0]
    out, n, ledger = wp.redistribute_v2_vacated_usage(frame, {'Out WR': {'plays_probability': 0.0}})
    recips = [r for entry in ledger for r in entry.get('recipients', [])]
    assert recips, 'expected at least one fill-in recipient'
    assert all('team_rank' in r for r in recips)
    ranks = {r['player']: r['team_rank'] for r in recips}
    if 'WR Big' in ranks and 'WR Small' in ranks:
        assert ranks['WR Big'] < ranks['WR Small']
