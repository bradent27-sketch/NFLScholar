"""
Offline tests for the cross-source player-name match keys
(data.utils.clean_name_exact / clean_name_for_merge).

These matter more than they look. Every failure mode they cover is SILENT:
a name that doesn't match doesn't raise, it just produces a player whose
snap share is blank, whose PFF grade is missing, or whose board row never
finds its market line - and the app renders perfectly while doing it. Two
of the three defect classes below were found by scanning the real 2025
files, and had been quietly dropping data for top-of-the-league players.

The negative tests are the important half. Widening a name match is easy;
the failure it introduces (one player's stats attributed to another) is
worse than the missing row it fixes, and equally silent.

Runs two ways: `python tests/test_player_names.py` needs nothing but the
app's own dependencies, and `pytest tests/` works if pytest is installed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.utils import (  # noqa: E402
    PLAYER_NAME_ALIASES, build_last_name_index, clean_name_exact,
    clean_name_for_merge, match_abbreviated_name,
)


def key(name):
    return clean_name_exact(pd.Series([name])).iloc[0]


def loose(name):
    return clean_name_for_merge(pd.Series([name])).iloc[0]


# --- the reported case ------------------------------------------------

def test_ken_and_kenneth_walker_share_an_exact_key():
    # snap_counts_2025.csv.csv says "Ken Walker III"; the weekly stats,
    # rosters and every PFF export say "Kenneth Walker III".
    assert key('Ken Walker III') == key('Kenneth Walker III')


def test_kenneth_walker_matches_with_the_suffix_dropped():
    # A source that drops BOTH the suffix and the long first name still has
    # to land on him - this is the tier-2 half of the fix.
    assert loose('Ken Walker') == loose('Kenneth Walker III')
    assert loose('Kenneth Walker') == loose('Ken Walker III')


def test_walker_does_not_collide_with_another_walker():
    for other in ('Austin Walker', 'Cody Walker', 'Mykel Walker'):
        assert key(other) != key('Kenneth Walker III'), other


# --- HTML entities (found in the real snap-count export) ---------------

def test_html_escaped_apostrophes_resolve():
    # 17 players in snap_counts_2025.csv.csv are stored this way. Stripping
    # punctuation without unescaping first makes it worse, not better:
    # "D&apos;Andre Swift" collapses to "damposandreswift".
    for escaped, plain in (
        ('Ja&apos;Marr Chase', "Ja'Marr Chase"),
        ('D&apos;Andre Swift', "D'Andre Swift"),
        ('Wan&apos;Dale Robinson', "Wan'Dale Robinson"),
        ('Adoree&apos; Jackson', "Adoree' Jackson"),
        ('Za&apos;Darius Smith', "Za'Darius Smith"),
    ):
        assert key(escaped) == key(plain), escaped


def test_entity_name_has_no_entity_debris_left_in_the_key():
    assert key('Ja&apos;Marr Chase') == 'jamarrchase'


# --- accents ----------------------------------------------------------

def test_accents_fold_rather_than_being_deleted():
    # [^a-z] deletes an accented letter outright, so "Jevón" became "jevn".
    assert key('Jevón Holland') == key('Jevon Holland') == 'jevonholland'
    assert key('Audric Estimé') == key('Audric Estime')
    assert loose('Rakeem Nuñez-Roches Sr.') == loose('Rakeem Nunez-Roches')


# --- the existing two-tier contract must still hold --------------------

def test_suffix_still_separates_two_real_players_at_tier_one():
    # Byron Murphy (Vikings CB) and Byron Murphy II (Seahawks DT) are two
    # people. Tier 1 is what keeps them apart; only tier 2 may merge them.
    assert key('Byron Murphy') != key('Byron Murphy II')
    assert loose('Byron Murphy') == loose('Byron Murphy II')


def test_plain_names_are_unchanged_by_the_alias_layer():
    assert key('Justin Jefferson') == 'justinjefferson'
    assert loose('A.J. Terrell Jr.') == loose('A.J. Terrell') == 'ajterrell'


# --- guard rails on the alias table ------------------------------------

def test_aliases_are_symmetric():
    # Registered both directions, so it doesn't matter which spelling a
    # future source arrives with.
    for variant, canonical in PLAYER_NAME_ALIASES:
        assert key(variant) == key(canonical), (variant, canonical)


def test_alias_table_never_merges_two_distinct_players():
    # Every canonical key must be reachable from exactly one real person.
    # A pair added carelessly (say "Mike Williams"/"Michael Williams", who
    # are different players) would show up here as one key with two
    # unrelated last-name groups behind it.
    by_key = {}
    for variant, canonical in PLAYER_NAME_ALIASES:
        for name in (variant, canonical):
            last = name.replace('.', ' ').split()
            last = [p for p in last if p.lower() not in {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}]
            by_key.setdefault(key(name), set()).add(last[-1].lower())
    for k, last_names in by_key.items():
        assert len(last_names) == 1, (k, last_names)


def test_first_initial_matching_is_not_performed():
    # Measured against the real 2025 files, an initial+last fallback
    # resolved "Landon Jackson" onto Lamar Jackson, "Terrell Edmunds" onto
    # Tremaine Edmunds and "Jeff Wilson Jr." onto Jared Wilson. Attributing
    # one player's line to another is worse than the blank it replaces, so
    # these must stay apart on both tiers.
    for a, b in (('Landon Jackson', 'Lamar Jackson'),
                 ('Terrell Edmunds', 'Tremaine Edmunds'),
                 ('Jeff Wilson Jr.', 'Jared Wilson'),
                 ('Carlos Washington Jr.', 'Casey Washington'),
                 ('Kalen King', 'Kobe King')):
        assert key(a) != key(b), (a, b)
        assert loose(a) != loose(b), (a, b)


# --- the F.Last bridge picks the repairs up too ------------------------

def test_abbreviated_match_survives_an_accent():
    index = build_last_name_index(['jevón holland', 'jaylen waddle'])
    assert match_abbreviated_name('J.Holland', index) == 'jevón holland'


def test_abbreviated_match_still_distinguishes_same_initial_teammates():
    # nflverse lengthens the prefix precisely to separate these two.
    index = build_last_name_index(['michael wilson', 'mack wilson'])
    assert match_abbreviated_name('Mi.Wilson', index) == 'michael wilson'
    assert match_abbreviated_name('Ma.Wilson', index) == 'mack wilson'


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
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


if __name__ == '__main__':
    sys.exit(main())
