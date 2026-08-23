"""Regression contracts for the team-constrained pass-capacity allocator.

Same posture as tests/test_rb_role_allocator_v2.py: target the small,
data-frame-only seam in data.pass_capacity_allocator directly, not the full
weekly projection build. The football contracts under test:

* a team's trusted top-N pass catchers keep their own projected value when
  the team's realistic budget can support it - the audit found that range
  was already accurate and there is no reason to refit a good number;
* the long tail beyond that shrinks to whatever budget remains, rather than
  each drawing an independent league-average share;
* receptions/receiving_yards/receiving_tds rescale by the exact same factor
  as targets, so a player's own catch rate and yards-per-target survive; and
* a team with no live QB volume and no prior-season history is left
  unmodified with an explicit ledger reason, never silently zeroed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.pass_capacity_allocator as pca  # noqa: E402


def _board_row(team, player, pos, targets, receptions=None, receiving_yards=None,
               receiving_tds=None, passing_attempts=0.0):
    return {
        'Team': team, 'Player': player, 'Pos': pos, 'targets': targets,
        'receptions': receptions if receptions is not None else targets * 0.65,
        'receiving_yards': receiving_yards if receiving_yards is not None else targets * 8.0,
        'receiving_tds': receiving_tds if receiving_tds is not None else targets * 0.05,
        'passing_attempts': passing_attempts,
    }


def _prior_history_row(team, week, passing_attempts, targets):
    return {'game_team': team, 'week': week, 'passing_attempts': passing_attempts, 'targets': targets}


def _approx(actual, expected, rel=1e-6):
    if expected == 0:
        return abs(actual) < 1e-9
    return abs(actual - expected) <= abs(expected) * rel + 1e-9


def test_trusted_tier_keeps_its_value_when_budget_allows():
    # A realistic team: 6 real pass catchers (already close to what a team
    # actually throws) plus 6 deep-bench names each drawing a small nonzero
    # target value from the flawed league-average fallback the audit found.
    rows = [_board_row('KC', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=34.0)]
    trusted_targets = [8.5, 7.0, 5.5, 4.0, 2.5, 1.8]
    for i, t in enumerate(trusted_targets):
        rows.append(_board_row('KC', f'Trusted{i}', 'WR' if i % 2 == 0 else 'TE', t))
    tail_targets = [1.2, 1.1, 0.9, 0.8, 0.6, 0.5]
    for i, t in enumerate(tail_targets):
        rows.append(_board_row('KC', f'Tail{i}', 'WR', t))
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=6)

    assert len(ledger) == 1
    entry = ledger.iloc[0]
    assert entry['capacity_source'] == 'live projected team pass attempts'
    trusted_rows = out[out['Player'].str.startswith('Trusted')]
    for i, t in enumerate(trusted_targets):
        assert _approx(trusted_rows.iloc[i]['targets'], t)
    tail_rows = out[out['Player'].str.startswith('Tail')]
    # The tail's total claim comfortably exceeds what's left after the
    # trusted tier, so every tail player must shrink from his input value.
    assert tail_rows['targets'].sum() < sum(tail_targets)
    assert (tail_rows['targets'] < pd.Series(tail_targets).to_numpy()).all()


def test_dependent_stats_rescale_by_the_same_factor_as_targets():
    # Capacity deliberately tight (20 attempts) so the bench tail's combined
    # claim (13.5) exceeds what's left after the one trusted player (9.0),
    # forcing an actual shrink - this test protects the RESCALE FACTOR
    # matching between targets and its dependents, not the direction.
    rows = [_board_row('SF', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=20.0)]
    # One trusted, high-volume tail player with a distinctive personal catch
    # rate and yards/target - if the rescale is done correctly, that shape
    # survives even though his target total shrinks.
    rows.append(_board_row('SF', 'WR1', 'WR', 9.0, receptions=7.0, receiving_yards=110.0, receiving_tds=0.8))
    for i in range(9):
        rows.append(_board_row('SF', f'Bench{i}', 'WR', 1.5, receptions=1.0, receiving_yards=12.0, receiving_tds=0.05))
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=1)

    bench = out[out['Player'].eq('Bench0')].iloc[0]
    factor = bench['targets'] / 1.5
    assert factor < 1.0  # the tail must have shrunk
    # rel is loose here on purpose: targets and each dependent are rounded
    # independently to 3 decimals, so re-deriving "factor" from the rounded
    # targets output picks up a little noise the un-rounded internal
    # computation didn't have. The contract under test is "same factor
    # applied", not bit-exact rounding agreement.
    assert _approx(bench['receptions'], 1.0 * factor, rel=1e-2)
    assert _approx(bench['receiving_yards'], 12.0 * factor, rel=1e-2)
    assert _approx(bench['receiving_tds'], 0.05 * factor, rel=1e-2)


def test_falls_back_to_prior_season_history_when_no_live_qb_attempts():
    # QB room unresolved this week (0 live attempts) - the team must still
    # get a sane, nonzero budget from its own prior-season team-games.
    rows = [_board_row('DAL', 'QB_unresolved', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=0.0)]
    for i in range(5):
        rows.append(_board_row('DAL', f'WR{i}', 'WR', 4.0))
    board = pd.DataFrame(rows)
    prior_history = pd.DataFrame([
        _prior_history_row('DAL', w, passing_attempts=32.0, targets=30.0) for w in range(1, 6)
    ])

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=prior_history, tier_size=8)

    entry = ledger.iloc[0]
    assert entry['capacity_source'] == 'prior-season team-game targets (no live QB attempts)'
    assert _approx(entry['capacity'], 30.0, rel=0.05)


def test_no_capacity_signal_leaves_team_unmodified():
    rows = [_board_row('LA', 'QB_unresolved', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=0.0)]
    for i in range(3):
        rows.append(_board_row('LA', f'WR{i}', 'WR', 3.0))
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=8)

    entry = ledger.iloc[0]
    assert entry['capacity_source'] == 'no capacity signal'
    wr_rows = out[out['Player'].str.startswith('WR')]
    assert (wr_rows['targets'] == 3.0).all()


def test_trusted_tier_alone_over_budget_scales_down_and_zeros_tail():
    # An extreme, deliberately pathological input: even the "trusted" top
    # tier alone claims more than the team could possibly throw.
    rows = [_board_row('CHI', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=20.0)]
    for i in range(3):
        rows.append(_board_row('CHI', f'Trusted{i}', 'WR', 15.0))
    rows.append(_board_row('CHI', 'Tail0', 'WR', 2.0))
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=3)

    entry = ledger.iloc[0]
    assert 'scaled down' in entry['reason']
    tail_row = out[out['Player'].eq('Tail0')].iloc[0]
    assert tail_row['targets'] == 0.0
    trusted_total = out[out['Player'].str.startswith('Trusted')]['targets'].sum()
    assert _approx(trusted_total, entry['capacity'], rel=0.01)


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as exc:  # pragma: no cover - standalone test report
            failures.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
