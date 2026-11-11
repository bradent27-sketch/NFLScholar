"""The props archive reader (2026-11-10, audit D2): an unreadable snapshot is skipped, not fatal. The reader moved to
data/prop_record.py on 2026-11-11 (B3); scripts/eval_posted_line_record.py keeps the old name."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import prop_record as pr  # noqa: E402
from scripts import eval_posted_line_record as epl  # noqa: E402


def _snapshot(week, stamp, player):
    return {'fetched_at': f'2026-11-0{week - 6}T12:00:00+00:00', 'season': 2026, 'week': week, 'status': {},
            'rows': [{'provider': 'Pinnacle', 'player': player, 'market': 'receiving_yards', 'scorable': True,
                      'period': 'game', 'line': 55.5, 'p_over': 0.5}]}


def test_load_archive_skips_a_truncated_snapshot(tmp_path, monkeypatch, capsys):
    good = json.dumps(_snapshot(8, 'a', 'Good Receiver'))
    (tmp_path / '2026_wk08_20261101T120000Z.json').write_text(good, encoding='utf-8')
    (tmp_path / '2026_wk09_20261103T001625Z.json').write_text(good[: len(good) // 2], encoding='utf-8')  # cut off
    (tmp_path / '2026_wk09_20261104T120000Z.json').write_text('[1, 2]', encoding='utf-8')               # not a snapshot
    monkeypatch.setattr(pr, 'ARCHIVE', str(tmp_path))

    arc = epl.load_archive(2026)

    assert list(arc['week']) == [8] and arc['line'].iloc[0] == 55.5
    assert arc.attrs['skipped_files'] == ['2026_wk09_20261103T001625Z.json', '2026_wk09_20261104T120000Z.json']
    assert '2026_wk09_20261103T001625Z.json' in capsys.readouterr().err
    assert len(os.listdir(tmp_path)) == 3                     # the bad files are left in place
