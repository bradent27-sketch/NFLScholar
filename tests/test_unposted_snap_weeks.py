"""mask_unposted_snap_weeks (2026-10-26): a played week whose snap counts are not posted yet reads as unknown, not 0%."""
import numpy as np
import pandas as pd

import data.loaders as loaders
from data.weekly_projections import expected_snap_share, mask_unposted_snap_weeks


def _stats():
    return pd.DataFrame({
        'player_display_name': ['A'] * 4 + ['B'] * 4,
        'team': ['NYJ'] * 4 + ['CHI'] * 4,
        'week': [4, 5, 6, 7] * 2,
        'weekly_snap_pct': [96, 95, 97, 0, 94, 95, 96, 95],
    })


def _snaps():
    rows = [(t, w) for t in ('NYJ', 'CHI') for w in (4, 5, 6)] + [('CHI', 7)]
    return pd.DataFrame({'team': [t for t, _ in rows], 'week': [w for _, w in rows], 'game_type': 'REG'})


def test_unposted_week_is_unknown_not_zero(monkeypatch):
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: _snaps())
    out, missing = mask_unposted_snap_weeks(_stats(), 2026, 'team')
    assert missing == [('NYJ', 7)]
    assert np.isnan(out.loc[3, 'weekly_snap_pct']) and out.loc[7, 'weekly_snap_pct'] == 95
    share = expected_snap_share(out, 'player_display_name', 'team', as_of_week=8)
    assert np.isclose(share['A'], 0.96) and np.isclose(share['B'], 0.95)
    before = expected_snap_share(_stats(), 'player_display_name', 'team', as_of_week=8)
    assert before['A'] < 0.75


def test_nothing_to_mask_or_no_source(monkeypatch):
    full = pd.concat([_snaps(), pd.DataFrame({'team': ['NYJ'], 'week': [7], 'game_type': ['REG']})], ignore_index=True)
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: full)
    out, missing = mask_unposted_snap_weeks(_stats(), 2026, 'team')
    assert missing == [] and out.loc[3, 'weekly_snap_pct'] == 0
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: pd.DataFrame())
    out, missing = mask_unposted_snap_weeks(_stats(), 2026, 'team')
    assert missing == [] and out.loc[3, 'weekly_snap_pct'] == 0


def test_missing_team_values_do_not_break_it(monkeypatch):
    # live 2026 stats carry a few rows with no team (categorical NaN); the build crashed sorting them (2026-10-26)
    monkeypatch.setattr(loaders, '_load_snap_counts_raw', lambda year: _snaps())
    stats = _stats()
    stats['team'] = pd.Categorical(stats['team'].where(stats.index != 0, None))
    out, missing = mask_unposted_snap_weeks(stats, 2026, 'team')
    assert missing == [('NYJ', 7)] and out.loc[0, 'weekly_snap_pct'] == 96
