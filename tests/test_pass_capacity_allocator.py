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


def _prior_position_history_row(team, week, position, targets, passing_attempts=0.0):
    return {'game_team': team, 'week': week, 'position': position,
            'targets': targets, 'passing_attempts': passing_attempts}


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


def test_trusted_tier_alone_over_budget_scales_everyone_proportionally():
    # An extreme input where even the "trusted" top tier alone claims more
    # than the team could possibly throw. This used to zero every player
    # outside the trusted tier outright - found live on 2026-08-24 to hard-
    # zero a real, established RB's (Derrick Henry / Kenneth Walker-shaped)
    # modest-but-genuine receiving role whenever a run-heavy team's trusted
    # tier was already exhausted by its WR/TE names alone. The fix: scale
    # EVERY catcher (trusted and tail) down by the same factor instead, so a
    # real tail player degrades proportionally rather than vanishing.
    rows = [_board_row('CHI', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=20.0)]
    for i in range(3):
        rows.append(_board_row('CHI', f'Trusted{i}', 'WR', 15.0))
    rows.append(_board_row('CHI', 'Tail0', 'WR', 2.0))
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=3)

    entry = ledger.iloc[0]
    assert 'scaled down' in entry['reason']
    current_total = 3 * 15.0 + 2.0
    factor = entry['capacity'] / current_total
    tail_row = out[out['Player'].eq('Tail0')].iloc[0]
    assert tail_row['targets'] > 0.0
    assert _approx(tail_row['targets'], 2.0 * factor, rel=1e-2)
    trusted_rows = out[out['Player'].str.startswith('Trusted')]
    for _, row in trusted_rows.iterrows():
        assert _approx(row['targets'], 15.0 * factor, rel=1e-2)
    allocated_total = trusted_rows['targets'].sum() + tail_row['targets']
    assert _approx(allocated_total, entry['capacity'], rel=0.01)


def test_rb_targets_are_unaffected_by_a_wr_te_only_roster_change():
    """The core defect this split fixes (explicit 2026-08-24 request): a
    WR/TE personnel change - a new signing, an injury, a rookie promoted -
    must never move a running back's conserved target value. Same QB
    attempts, same RB row, same (absent) prior history; only the WR/TE rows
    differ between the two boards."""
    def _board(wr_te_rows):
        rows = [_board_row('SEA', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=34.0),
               _board_row('SEA', 'Lead RB', 'RB', 4.0)]
        rows.extend(wr_te_rows)
        return pd.DataFrame(rows)

    modest_wr_te = [_board_row('SEA', 'WR1', 'WR', 7.0), _board_row('SEA', 'WR2', 'WR', 5.0)]
    busy_wr_te = [
        _board_row('SEA', 'WR1', 'WR', 9.0), _board_row('SEA', 'WR2', 'WR', 7.0),
        _board_row('SEA', 'NewSigning', 'WR', 6.0), _board_row('SEA', 'TE1', 'TE', 5.0),
        _board_row('SEA', 'RookieWR', 'WR', 3.0),
    ]

    out_modest, _ = pca.apply_pass_capacity_conservation(_board(modest_wr_te), prior_history=None, tier_size=8)
    out_busy, _ = pca.apply_pass_capacity_conservation(_board(busy_wr_te), prior_history=None, tier_size=8)

    rb_modest = out_modest[out_modest['Player'].eq('Lead RB')].iloc[0]['targets']
    rb_busy = out_busy[out_busy['Player'].eq('Lead RB')].iloc[0]['targets']
    assert _approx(rb_modest, rb_busy)
    # Not just coincidentally equal - the WR/TE side DID move in response to
    # its own roster change, so this test would have caught the original
    # defect (a shared trusted/tail pool moving the RB's factor too).
    wr1_modest = out_modest[out_modest['Player'].eq('WR1')].iloc[0]['targets']
    wr1_busy = out_busy[out_busy['Player'].eq('WR1')].iloc[0]['targets']
    assert not _approx(wr1_modest, wr1_busy)


def test_rb_group_is_still_scaled_to_its_own_sub_budget_when_it_over_claims():
    """"A slight ding in some cases" is still expected - explicit ask - when
    the RB group's OWN claim exceeds ITS OWN sub-budget. Two backs each
    drawing implausibly high receiving volume must shrink using the RB-only
    trusted tier (2) and RB-only sub-budget, while the untouched WR/TE room
    on the same team proves the two groups are fit independently."""
    rows = [
        _board_row('MIA', 'QB1', 'QB', 0.0, 0.0, 0.0, 0.0, passing_attempts=30.0),
        _board_row('MIA', 'RB1', 'RB', 12.0),
        _board_row('MIA', 'RB2', 'RB', 10.0),
        _board_row('MIA', 'WR1', 'WR', 6.0),
        _board_row('MIA', 'WR2', 'WR', 5.0),
    ]
    board = pd.DataFrame(rows)

    out, ledger = pca.apply_pass_capacity_conservation(board, prior_history=None, tier_size=8)

    rb_rows = out[out['Player'].str.startswith('RB')]
    assert rb_rows['targets'].sum() < 22.0
    assert (rb_rows['targets'] < pd.Series([12.0, 10.0]).to_numpy()).all()
    rb_entry = ledger[ledger['position_group'].eq('RB')].iloc[0]
    assert rb_entry['trusted_count'] == 2
    assert 'scaled down' in rb_entry['reason']
    # The WR/TE room's own budget was never touched by RB's overflow.
    wr_rows = out[out['Player'].str.startswith('WR')]
    assert _approx(wr_rows['targets'].sum(), 11.0)


def test_derive_team_rb_catcher_share_reads_real_history_and_falls_back_league_wide():
    history = pd.DataFrame(
        [_prior_position_history_row('BUF', w, 'RB', targets=6.0) for w in range(1, 5)]
        + [_prior_position_history_row('BUF', w, 'WR', targets=20.0) for w in range(1, 5)]
        + [_prior_position_history_row('BUF', w, 'TE', targets=4.0) for w in range(1, 5)]
    )
    team_share, league_share = pca.derive_team_rb_catcher_share(history)
    # 6 / (6 + 20 + 4) = 0.2 for every one of BUF's team-games.
    assert _approx(team_share['BUF'], 0.2, rel=1e-6)
    assert _approx(league_share, 0.2, rel=1e-6)

    empty_team_share, empty_league_share = pca.derive_team_rb_catcher_share(pd.DataFrame())
    assert empty_team_share == {}
    assert _approx(empty_league_share, pca.FALLBACK_RB_CATCHER_SHARE)


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
