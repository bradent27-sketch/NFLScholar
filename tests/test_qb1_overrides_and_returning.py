import numpy as np
import pandas as pd
import pytest

import data.weekly_projections as wp


# --- dated QB1 overrides --------------------------------------------------------

def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_legacy_three_column_file_still_reads_as_season_long(tmp_path):
    path = tmp_path / 'qb1.csv'
    _write(path, [{'year': 2026, 'team': 'CHI', 'player': 'Caleb Williams'}])
    table, problem = wp.load_qb1_overrides(2026, path=path)
    assert problem is None
    for week in (1, 9, 18):
        assert wp.qb1_overrides_for_week(table, week)['player'].tolist() == ['Caleb Williams']


def test_week_windows_pick_the_row_in_force(tmp_path):
    path = tmp_path / 'qb1.csv'
    _write(path, [{'year': 2026, 'team': 'WAS', 'player': 'Jayden Daniels', 'from_week': '', 'through_week': 2},
                  {'year': 2026, 'team': 'WAS', 'player': 'Marcus Mariota', 'from_week': 3, 'through_week': ''}])
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert wp.qb1_overrides_for_week(table, 1)['player'].tolist() == ['Jayden Daniels']
    assert wp.qb1_overrides_for_week(table, 2)['player'].tolist() == ['Jayden Daniels']
    assert wp.qb1_overrides_for_week(table, 3)['player'].tolist() == ['Marcus Mariota']
    assert wp.qb1_overrides_for_week(table, 12)['player'].tolist() == ['Marcus Mariota']


def test_save_with_week_closes_the_previous_choice_instead_of_overwriting_it(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, '_invalidate_weekly_projection_cache', lambda: None)
    path = tmp_path / 'qb1.csv'
    _write(path, [{'year': 2026, 'team': 'CHI', 'player': 'Caleb Williams'}])
    wp.save_qb1_override(2026, 'CHI', 'Case Keenum', path=path, week=3)
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert wp.qb1_overrides_for_week(table, 2)['player'].tolist() == ['Caleb Williams']
    assert wp.qb1_overrides_for_week(table, 3)['player'].tolist() == ['Case Keenum']
    # A second save in a later week stacks another window; earlier weeks are untouched.
    wp.save_qb1_override(2026, 'CHI', 'Tyson Bagent', path=path, week=4)
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert [wp.qb1_overrides_for_week(table, w)['player'].tolist() for w in (2, 3, 4, 9)] == [
        ['Caleb Williams'], ['Case Keenum'], ['Tyson Bagent'], ['Tyson Bagent']]
    # Re-saving the same week replaces that week's choice rather than duplicating it.
    wp.save_qb1_override(2026, 'CHI', 'Caleb Williams', path=path, week=4)
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert wp.qb1_overrides_for_week(table, 4)['player'].tolist() == ['Caleb Williams']
    assert wp.qb1_overrides_for_week(table, 3)['player'].tolist() == ['Case Keenum']


def test_save_through_week_and_clear_from_a_week(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, '_invalidate_weekly_projection_cache', lambda: None)
    path = tmp_path / 'qb1.csv'
    wp.save_qb1_override(2026, 'TB', 'Baker Mayfield', path=path, week=1)
    wp.save_qb1_override(2026, 'TB', 'Jalon Daniels', path=path, week=4, through_week=4)
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert wp.qb1_overrides_for_week(table, 4)['player'].tolist() == ['Jalon Daniels']
    assert wp.qb1_overrides_for_week(table, 5).empty     # the one-week choice expired, nothing reopened
    assert wp.qb1_overrides_for_week(table, 3)['player'].tolist() == ['Baker Mayfield']
    wp.clear_qb1_override(2026, 'TB', path=path, week=2)
    table, _ = wp.load_qb1_overrides(2026, path=path)
    assert wp.qb1_overrides_for_week(table, 1)['player'].tolist() == ['Baker Mayfield']
    assert wp.qb1_overrides_for_week(table, 2).empty and wp.qb1_overrides_for_week(table, 4).empty


def test_shipped_override_file_is_dated_and_unambiguous():
    table, problem = wp.load_qb1_overrides(2026)
    assert problem is None
    for week in range(1, 19):
        in_force = wp.qb1_overrides_for_week(table, week)
        assert not in_force['team'].duplicated().any(), f'two choices for one team in week {week}'


# --- returning starter -----------------------------------------------------------

def _history(rows):
    out = pd.DataFrame(rows, columns=['name', 'team', 'week', 'weekly_snap_pct'])
    out['position'] = 'QB'
    out['has_snap_match'] = True
    return out


def _resolve(history, as_of_week, out_by_week, prior_share=None, unavailable=(), overrides=None, roster=None):
    current = history.sort_values('week').drop_duplicates('name', keep='last')[['name', 'team']]
    returning = {'out_by_week': out_by_week, 'prior_share': prior_share or {},
                 'roster': roster if roster is not None else pd.DataFrame(columns=['_team', '_key', '_player'])}
    kw = dict(overrides=overrides if overrides is not None else pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS),
              unavailable_players=set(unavailable))
    base = wp.resolve_inseason_qb1s(current, 'name', 'team', history, 'name', 'team', as_of_week, 2026, **kw)
    flag = wp.resolve_inseason_qb1s(current, 'name', 'team', history, 'name', 'team', as_of_week, 2026,
                                    returning=returning, **kw)
    return base, flag


def key(name):
    return wp.clean_name_exact(pd.Series([name])).iloc[0]


def test_injured_week1_starter_reclaims_the_job_when_off_the_report():
    hist = _history([['Starter', 'SEA', 1, 100], ['Fill In', 'SEA', 2, 100], ['Fill In', 'SEA', 3, 100]])
    base, flag = _resolve(hist, 4, {1: set(), 2: {key('Starter')}, 3: {key('Starter')}})
    assert base['by_team']['SEA']['player'] == 'Fill In'
    assert flag['by_team']['SEA']['player'] == 'Starter'
    assert flag['by_team']['SEA']['status'] == 'returning_starter'


def test_starter_who_was_healthy_and_did_not_play_has_lost_the_job():
    hist = _history([['Starter', 'NO', 1, 100], ['Fill In', 'NO', 2, 100], ['Fill In', 'NO', 3, 100]])
    _, flag = _resolve(hist, 4, {2: {key('Starter')}, 3: set()})   # healthy in week 3, still sat
    assert flag['by_team']['NO']['player'] == 'Fill In'


def test_a_fill_in_who_got_hurt_is_not_restored_over_the_current_starter():
    hist = _history([['Starter', 'MIA', 1, 100], ['Fill In', 'MIA', 2, 100],
                     ['Third', 'MIA', 3, 100], ['Third', 'MIA', 4, 100]])
    both = {key('Fill In'), key('Starter')}
    _, flag = _resolve(hist, 5, {2: {key('Starter')}, 3: both, 4: both}, unavailable={'Starter'})
    assert flag['by_team']['MIA']['player'] == 'Third'


def test_still_unavailable_starter_is_not_picked():
    hist = _history([['Starter', 'CIN', 1, 100], ['Fill In', 'CIN', 2, 100]])
    _, flag = _resolve(hist, 3, {2: {key('Starter')}}, unavailable={'Starter'})
    assert flag['by_team']['CIN']['player'] == 'Fill In'


def test_last_years_starter_with_no_snaps_yet_comes_from_the_roster():
    hist = _history([['Fill In', 'ATL', 1, 100], ['Fill In', 'ATL', 2, 100]])
    roster = pd.DataFrame({'_team': ['ATL'], '_key': [key('Prior Starter')], '_player': ['Prior Starter']})
    _, flag = _resolve(hist, 3, {1: {key('Prior Starter')}, 2: {key('Prior Starter')}},
                       prior_share={('ATL', key('Prior Starter')): 0.9}, roster=roster)
    assert flag['by_team']['ATL']['player'] == 'Prior Starter'
    # Last year's starter for ANOTHER team does not count.
    _, flag = _resolve(hist, 3, {1: {key('Prior Starter')}, 2: {key('Prior Starter')}},
                       prior_share={('MIA', key('Prior Starter')): 0.9}, roster=roster)
    assert flag['by_team']['ATL']['player'] == 'Fill In'


def test_override_for_a_player_listed_out_is_skipped_for_that_week():
    hist = _history([['Starter', 'MIN', 1, 100], ['Backup', 'MIN', 2, 100]])
    overrides = pd.DataFrame([{'year': 2026, 'team': 'MIN', 'player': 'Starter', 'from_week': np.nan}])
    base, flag = _resolve(hist, 3, {2: {key('Starter')}}, unavailable={'Starter'}, overrides=overrides)
    assert base['by_team']['MIN']['status'] == 'manual_override'          # legacy: the out player kept QB1
    assert flag['by_team']['MIN']['player'] == 'Backup'
    assert any('listed out/doubtful this week' in w for w in flag['warnings'])


def test_fill_in_override_set_during_the_injury_gives_way_when_the_starter_returns():
    hist = _history([['Starter', 'CHI', 1, 100], ['Starter', 'CHI', 2, 100],
                     ['Fill In', 'CHI', 3, 100], ['Fill In', 'CHI', 4, 100]])
    overrides = pd.DataFrame([{'year': 2026, 'team': 'CHI', 'player': 'Fill In', 'from_week': 3.0}])
    _, flag = _resolve(hist, 5, {3: {key('Starter')}, 4: {key('Starter')}}, overrides=overrides)
    assert flag['by_team']['CHI']['player'] == 'Starter'
    assert any('was set in week 3 while Starter was injured' in w for w in flag['warnings'])
    # An override saved BEFORE the injury is a deliberate choice, not a stale fill-in, and wins.
    overrides = pd.DataFrame([{'year': 2026, 'team': 'CHI', 'player': 'Fill In', 'from_week': 1.0}])
    _, flag = _resolve(hist, 5, {3: {key('Starter')}, 4: {key('Starter')}}, overrides=overrides)
    assert flag['by_team']['CHI']['status'] == 'manual_override'


# --- the rule only runs once this week's availability report is loaded ------------

def test_returning_rule_is_gated_on_a_loaded_availability_report():
    flag = frozenset({'v2_qb1_returning_starter'})
    # Report loaded (a replay always has one): the rule runs, no warning.
    assert wp.qb1_returning_gate(flag, {'someone': {'plays_probability': 0.0}}, 5) == (True, None)
    # Flag off: nothing to say either way.
    assert wp.qb1_returning_gate(frozenset(), {}, 5) == (False, None)
    # Flag on but no report yet: skipped, and the reason is surfaced.
    use, warning = wp.qb1_returning_gate(flag, {}, 5)
    assert use is False
    assert 'week 5' in warning and 'not applied' in warning
