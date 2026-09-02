"""
Offline tests for the consolidated import registry (data/import_status.py)
and the per-player Book Proj breakdown (data.odds_projections._explain_book_line).

Both exist to answer "where did this number come from", so both fail in the
same silent way: a status that says a file loaded when it didn't, or a
breakdown that omits the half of the projection that came from this app
rather than from a book, still renders perfectly and is still misleading.

Runs two ways: `python tests/test_imports_and_book_detail.py` needs nothing
but the app's own dependencies, and `pytest tests/` works if pytest is
installed.
"""
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

from data.import_status import (  # noqa: E402
    ALL_IMPORT_KEYS, IMPORT_GROUPS, format_age, format_size, status_table,
)
from data.odds_projections import _explain_book_line  # noqa: E402


# --- the registry -------------------------------------------------------

def test_every_import_appears_exactly_once():
    # The old shape repeated the uploader list in three corners of the
    # settings panel, which is how a source ends up with an uploader and no
    # status, or a status and no uploader.
    assert len(ALL_IMPORT_KEYS) == len(set(ALL_IMPORT_KEYS))
    table = status_table({})
    assert len(table) == len(ALL_IMPORT_KEYS)


def test_every_book_the_board_reads_has_a_row():
    from data.odds_sources import BOOKS
    keys = set(ALL_IMPORT_KEYS)
    for book in BOOKS:
        assert book in keys, book


def test_the_table_carries_the_columns_the_panel_renders():
    table = status_table({})
    for column in ('Source', 'Loaded', 'Imported', 'Age', 'Size',
                   'What came out of it', 'File'):
        assert column in table.columns, column


def test_a_missing_import_reads_as_not_loaded_with_no_timestamp(tmp_path, monkeypatch):
    # A blank timestamp, not today's date - "imported just now" for a file
    # that was never imported is the exact confusion this panel replaces.
    #
    # Runs against an EMPTY temp directory, not the developer's real
    # external_data/. This used to assert on 'Underdog' against live disk and
    # so failed on any machine that had actually saved an Underdog payload -
    # a false alarm about the app on a checkout where nothing was wrong
    # (fixed 2026-09-01). What is under test is the "never imported" 
    # rendering, which needs a guaranteed-absent source to be meaningful.
    import data.import_status as ist
    monkeypatch.chdir(tmp_path)
    for attr in ('EXTERNAL_DIR', 'EXTERNAL_DATA_DIR', 'DATA_DIR'):
        if hasattr(ist, attr):
            monkeypatch.setattr(ist, attr, tmp_path / 'external_data')

    table = status_table({}).set_index('Source')
    row = table.loc['Underdog']
    assert row['Loaded'] == '—', "an absent import must not render as loaded"
    assert row['Imported'] == ''
    assert row['Age'] == ''


def test_a_session_held_import_reports_its_rows_and_upload_time():
    from data.import_status import UPLOAD_TIME_KEYS
    from ui.tabs.draft_hq import ECR_UPLOAD_KEY
    frame = pd.DataFrame({'Player': ['A', 'B'], 'ECR': [1, 2], 'ECR SD': [0.5, None]})
    session = {ECR_UPLOAD_KEY: frame,
               UPLOAD_TIME_KEYS['ecr']: datetime.datetime.now()}
    row = status_table(session).set_index('Source').loc['FantasyPros rankings']
    assert row['Loaded'] == '✅'
    assert '2 players' in row['What came out of it']
    assert 'expert spread' in row['What came out of it']
    assert row['Imported'] != ''
    # Session-held imports have no file, and must say so rather than showing
    # a path that doesn't exist.
    assert 'session' in row['File']


def test_a_broken_source_is_reported_rather_than_raising():
    # The panel exists to make a broken import visible; taking the tab down
    # with it would be the one outcome worse than the silence it replaces.
    table = status_table({'not': 'a real session'})
    assert len(table) == len(ALL_IMPORT_KEYS)


def test_group_labels_are_stable_and_non_empty():
    # The UI renders one tab per group, in this order.
    groups = [group for group, _entries in IMPORT_GROUPS]
    assert groups == ['Rankings & ADP', 'Projections & notes', 'Sportsbooks (season-long)']
    for _group, entries in IMPORT_GROUPS:
        for key, label, purpose in entries:
            assert key and label and purpose


def test_age_and_size_formatting():
    now = datetime.datetime.now()
    assert format_age(None) == ''
    assert format_age(now) == 'just now'
    assert 'days ago' in format_age(now - datetime.timedelta(days=4))
    assert format_size(None) == ''
    assert format_size(500) == '500 B'
    assert format_size(1024 * 1024 * 3) == '3.0 MB'


# --- the book breakdown -------------------------------------------------

def row(**over):
    base = {'providers': 'PrizePicks, Underdog', 'Books': 2, 'n_lines': 6}
    base.update(over)
    return pd.Series(base)


def test_breakdown_names_the_books_and_counts_them():
    text = _explain_book_line(row(), [('receiving_yards', 1258.0, 1258.0)], [])
    assert 'PrizePicks, Underdog' in text
    assert '2 books' in text


def test_breakdown_shows_the_posted_line_and_the_corrected_value():
    # A line is a median, a projection is a mean, and for touchdowns the gap
    # is large. Showing only the corrected number makes the panel disagree
    # with the book's own page for no visible reason.
    text = _explain_book_line(row(), [('receiving_tds', 9.0, 9.5)], [])
    assert 'Rec TD 9' in text
    assert '9.5 used' in text


def test_no_correction_is_shown_when_the_number_did_not_move():
    text = _explain_book_line(row(), [('receiving_yards', 1258.0, 1258.0)], [])
    assert 'used' not in text.split('**Filled in')[0]


def test_breakdown_states_what_came_from_this_app_instead():
    # The whole point. Two players can carry the same Book Proj with
    # completely different provenance, and only this line separates them.
    text = _explain_book_line(
        row(providers='Underdog', Books=1, n_lines=2),
        [('passing_yards', 4225.5, 4225.5)],
        [('rushing_yards', 300.0), ('rushing_tds', 2.0)],
    )
    assert "Filled in from this app's projection" in text
    assert 'Rush yds 300.0' in text


def test_a_fully_priced_player_says_so_rather_than_going_silent():
    text = _explain_book_line(row(), [('receiving_yards', 1258.0, 1258.0)], [])
    assert 'the book priced it all' in text


def test_a_player_with_no_market_stats_says_nothing_was_priced():
    text = _explain_book_line(row(), [], [('rushing_yards', 900.0)])
    assert 'nothing priced' in text


def test_a_long_fallback_list_is_truncated_with_a_count():
    ours = [(f'stat_{i}', float(i)) for i in range(10)]
    text = _explain_book_line(row(), [], ours)
    assert '+4 more' in text


def test_missing_provider_metadata_degrades_rather_than_raising():
    bare = pd.Series({'providers': None, 'Books': None, 'n_lines': None})
    text = _explain_book_line(bare, [('receiving_yards', 100.0, 100.0)], [])
    assert 'unknown' in text
    assert 'nan' not in text.lower()


def test_breakdown_reaches_the_board_as_a_column():
    from data.odds_projections import attach_book_projection
    board = pd.DataFrame({'Player': ['A', 'B'], 'Proj Pts': [200.0, 150.0]})
    projection = pd.DataFrame({
        'board_player': ['A'], 'Book Proj': [210.0], 'Book Δ': [10.0],
        'Book Share': [0.8], 'Book Detail': ['**Books read:** Underdog'],
    })
    out = attach_book_projection(board, projection)
    assert out.loc[0, 'Book Detail'] == '**Books read:** Underdog'
    # Blank, not carried over from the previous row, for a player no book
    # priced.
    assert pd.isna(out.loc[1, 'Book Detail'])


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
