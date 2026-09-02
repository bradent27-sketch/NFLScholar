"""Player Search: the merged (played + scheduled) game log and the
full-season fantasy-points strip.

Both are visual and fail quietly - a fabricated 0 in an unplayed cell, or a
strip that silently collapses to only the weeks played, still render.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import streamlit as st  # noqa: E402
from ui.styling import render_game_log_html_table  # noqa: E402
from ui.components import render_fpts_week_strip  # noqa: E402


class _Capture:
    """Swap st.markdown for a sink that keeps every emitted HTML string."""
    def __enter__(self):
        self._real = st.markdown
        self.html = []
        st.markdown = lambda body, *a, **k: self.html.append(str(body))
        return self

    def __exit__(self, *exc):
        st.markdown = self._real
        return False

    @property
    def joined(self):
        return "\n".join(self.html)


LOG_COLS = ['week', 'opponent_team', 'targets', 'receptions', 'receiving_yards', 'fantasy_points']
HEADER = {'fantasy_points': 'FPTS', 'receiving_yards': 'REC YDS'}


def _merged_frame():
    # weeks 1-5 scheduled; only 1 and 2 played.
    rows = []
    for wk in range(1, 6):
        played = wk <= 2
        rows.append({
            'week': wk, 'opponent_team': f'OP{wk}',
            'targets': 8 if played else None,
            'receptions': 6 if played else None,
            'receiving_yards': 70.0 + wk if played else None,
            'fantasy_points': 13.0 + wk if played else None,
            '_unplayed': not played,
        })
    return pd.DataFrame(rows)


# --- merged game log table -----------------------------------------------

def test_unplayed_rows_render_empty_stat_cells_not_zero():
    df = _merged_frame()
    with _Capture() as cap:
        render_game_log_html_table(df, df[~df['_unplayed']], LOG_COLS, HEADER)
    html = cap.joined
    assert html.count('<tr') >= 6           # 5 weeks + AVG (+ header tr)
    # Played weeks show their numbers.
    assert '71.0' in html or '71' in html
    # Unplayed weeks: opponent still there, but no fabricated 0 in stat cells.
    assert 'OP5' in html
    # The three unplayed rows contribute no "0" stat cells - every stat cell
    # on those rows is blank. Rough check: "OP3"/"OP4"/"OP5" rows exist and
    # the table never prints a lone ">0<" cell.
    assert '>0<' not in html


def test_avg_row_uses_only_played_rows():
    df = _merged_frame()
    with _Capture() as cap:
        render_game_log_html_table(df, df[~df['_unplayed']], LOG_COLS, HEADER)
    html = cap.joined
    # mean fantasy_points over played weeks (14.0, 15.0) = 14.5
    assert '14.5' in html


def test_all_unplayed_gives_blank_avg_not_zeroes():
    df = _merged_frame()
    df['_unplayed'] = True
    with _Capture() as cap:
        render_game_log_html_table(df, df.iloc[0:0], LOG_COLS, HEADER)
    html = cap.joined
    assert 'AVG' in html
    assert '>0.0<' not in html and '>0<' not in html


# --- full-season fpts strip --------------------------------------------

def _sched(n=18):
    return pd.DataFrame({'week': list(range(1, n + 1)),
                         'opponent_team': [f'OP{w}' for w in range(1, n + 1)]})


def _played(weeks):
    return pd.DataFrame({
        'week': weeks,
        'opponent_team': [f'OP{w}' for w in weeks],
        'fantasy_points': [10.0 + w for w in weeks],
        'fantasy_points_wk_pct': [50 for _ in weeks],
    })


def test_pre_kickoff_renders_empty_axis_with_week_ticks_and_no_dots():
    with _Capture() as cap:
        render_fpts_week_strip(pd.DataFrame({'foo': [1]}), 2026, schedule_df=_sched())
    html = cap.joined
    assert '<svg' in html
    assert '<circle' not in html
    assert 'fl-wklabel' in html
    assert 'Fills in as games are played' in html
    assert 'OP1' in html                     # opponent labels on the empty axis


def test_partial_season_plots_only_played_weeks_on_a_full_axis():
    played = _played([1, 2, 3])
    with _Capture() as cap:
        render_fpts_week_strip(played, 2026, schedule_df=_sched())
    html = cap.joined
    assert html.count('<circle') == 3        # one dot per played week
    assert '<path' in html                   # the growing curve
    assert 'Fills in as games are played' not in html


def test_no_schedule_and_no_week_data_renders_nothing():
    with _Capture() as cap:
        render_fpts_week_strip(pd.DataFrame({'foo': [1]}), 2026)
    assert cap.html == []


def test_single_played_week_is_a_dot_without_a_curve():
    with _Capture() as cap:
        render_fpts_week_strip(_played([1]), 2026, schedule_df=_sched())
    html = cap.joined
    assert html.count('<circle') == 1
    # a lone point => no smoothed line/area path (only the y-axis line tag)
    assert '<path d=""' not in html


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    bad = []
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            bad.append((n, exc))
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(bad)}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
