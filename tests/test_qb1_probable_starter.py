"""v2_qb1_probable_starter_fallback (shipped 2026-10-20, backlog M6): a room the resolver leaves unresolved gets a
probability-weighted starter instead of every QB projected at zero."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.options.mode.string_storage = "python"

import data.weekly_projections as wp  # noqa: E402


def key(name):
    return wp.clean_name_exact(pd.Series([name])).iloc[0]


def _history(rows):
    out = pd.DataFrame(rows, columns=['name', 'team', 'week', 'weekly_snap_pct'])
    out['position'] = 'QB'
    out['has_snap_match'] = True
    return out


def _resolve(history, as_of_week, unavailable=(), fallback=True):
    current = history.sort_values('week').drop_duplicates('name', keep='last')[['name', 'team']]
    return wp.resolve_inseason_qb1s(
        current, 'name', 'team', history, 'name', 'team', as_of_week, 2026,
        overrides=pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS), unavailable_players=set(unavailable), fallback=fallback)


def test_flag_is_registered_and_shipped():
    assert 'v2_qb1_probable_starter_fallback' in wp.MODEL_FEATURES
    assert 'v2_qb1_probable_starter_fallback' in wp.DEFAULT_FEATURES


def test_off_by_default_leaves_the_room_unresolved():
    # 2026 PHI week 6: the starter is listed out and the only other QB has a single relief game
    hist = _history([['Starter', 'PHI', w, 100] for w in range(1, 6)] + [['Backup', 'PHI', 3, 12]])
    out = _resolve(hist, 6, unavailable={key('Starter')}, fallback=False)
    assert out['by_team']['PHI']['status'] == 'selection_required'
    assert out['selected'] == {} and out['weights'] == {}


def test_a_sole_available_qb_is_the_probable_starter_at_full_weight():
    hist = _history([['Starter', 'PHI', w, 100] for w in range(1, 6)] + [['Backup', 'PHI', 3, 12]])
    out = _resolve(hist, 6, unavailable={key('Starter')})
    assert out['by_team']['PHI']['status'] == 'probable_starter'
    assert out['selected'] == {('PHI', key('Backup')): 'probable_starter'}
    assert out['weights'] == {('PHI', key('Backup')): 1.0}
    assert 'PHI' in out['selection_required_teams']           # still asks for a confirmed choice


def test_two_candidates_are_weighted_by_games_played_then_recency():
    hist = _history([['Vet', 'KC', 1, 60], ['Vet', 'KC', 2, 55], ['Vet', 'KC', 3, 60], ['Young', 'KC', 3, 40],
                     ['Young', 'KC', 4, 45], ['Vet', 'KC', 4, 50]])
    out = _resolve(hist, 5)
    assert out['by_team']['KC']['status'] == 'probable_starter'
    w = {k[1]: v for k, v in out['weights'].items()}
    assert set(w) == {key('Vet'), key('Young')}
    assert w[key('Vet')] > w[key('Young')] and abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w[key('Vet')] - 0.62 / 0.97) < 1e-9


def test_a_clear_full_snap_starter_is_untouched():
    hist = _history([['Starter', 'BUF', 1, 100], ['Starter', 'BUF', 2, 100], ['Backup', 'BUF', 2, 5]])
    on, off = _resolve(hist, 3), _resolve(hist, 3, fallback=False)
    assert on['by_team']['BUF'] == off['by_team']['BUF'] and on['weights'] == {}


def test_a_qb_not_listed_out_is_required_and_a_room_with_no_one_stays_unresolved():
    hist = _history([['Only', 'NYJ', 1, 100], ['Only', 'NYJ', 2, 100]])
    out = _resolve(hist, 3, unavailable={key('Only')})
    assert out['by_team']['NYJ']['status'] == 'selection_required' and out['weights'] == {}


def test_a_qb_with_no_real_snap_game_is_not_a_candidate():
    hist = _history([['Starter', 'DEN', 1, 100], ['Ghost', 'DEN', 2, 0]])
    out = _resolve(hist, 3, unavailable={key('Starter')})
    assert out['by_team']['DEN']['status'] != 'probable_starter'


def test_empty_resolution_has_the_weights_key():
    out = wp.resolve_inseason_qb1s(pd.DataFrame(), 'name', 'team', pd.DataFrame(), 'name', 'team', 3, 2026)
    assert out['weights'] == {}
