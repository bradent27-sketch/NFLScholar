"""Regression contracts for the QB1 team/style volume blend.

Targets data.qb_volume_blend directly, same posture as the other V2
allocator test files. The football contracts under test:

* a QB1 with only a thin backup-relief sample gets his personal rush share
  shrunk hard toward the league mean, anchored to his TEAM's own dropback
  volume - not left at his raw small-sample per-game rate (the Malik Willis
  case the user described: a strong-rushing backup projected as a starter
  off 4 relief appearances);
* a QB1 with a full, established starter sample keeps a blend dominated by
  his own measured style, since there is enough evidence to trust it;
* dependent stats (yards, TDs, completions, INTs) preserve the player's own
  per-attempt/per-carry efficiency exactly, scaled onto the new blended
  volume - never a league or team efficiency; and
* a non-QB1 row and a QB1 with no prior-season data are both no-ops (NaN),
  since the blend has nothing to correct against.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.qb_volume_blend as qvb  # noqa: E402


def _approx(actual, expected, rel=1e-6):
    if expected == 0:
        return abs(actual) < 1e-9
    return abs(actual - expected) <= abs(expected) * rel + 1e-9


def _prior_row(team, player, week, passing_attempts, rushing_attempts,
               passing_yards=None, passing_tds=0.0, passing_completions=None,
               passing_interceptions=0.0, rushing_yards=None, rushing_tds=0.0):
    return {
        'game_team': team, 'player_display_name': player, 'position': 'QB', 'week': week,
        'passing_attempts': passing_attempts, 'rushing_attempts': rushing_attempts,
        'passing_yards': passing_yards if passing_yards is not None else passing_attempts * 7.0,
        'passing_tds': passing_tds,
        'passing_completions': passing_completions if passing_completions is not None else passing_attempts * 0.63,
        'passing_interceptions': passing_interceptions,
        'rushing_yards': rushing_yards if rushing_yards is not None else rushing_attempts * 5.5,
        'rushing_tds': rushing_tds,
    }


def _willis_like_prior_history():
    """4 backup relief appearances: high personal rush share, thin sample.

    Modeled on the real case: Malik Willis (TEN, 2025) - a few relief
    appearances with heavy rushing usage relative to his own dropbacks, on a
    team (his prior team) that itself ran a normal, much lower rush share.
    Distinct opponents so this reads as a real small sample, not one game
    duplicated four times.
    """
    rows = []
    for i, week in enumerate((3, 8, 14, 17)):
        rows.append(_prior_row('TEN', 'Malik Willis', week, passing_attempts=9.0, rushing_attempts=5.0))
    # A full season of a normal starter elsewhere in the league, so the
    # league-mean rush share this blend shrinks toward is not degenerate.
    for week in range(1, 18):
        rows.append(_prior_row('KC', 'Established Starter', week, passing_attempts=34.0, rushing_attempts=2.0))
    # The QB1's own NEW team's prior-season starter, establishing MIA's own
    # team dropback capacity distinct from Willis's own small sample.
    for week in range(1, 18):
        rows.append(_prior_row('MIA', 'Prior Team QB', week, passing_attempts=30.0, rushing_attempts=2.0))
    # A couple more ordinary full-season starters, so the league mean this
    # blend shrinks toward reflects a real ~35-QB league rather than being
    # skewed by Willis's own outlier rate being one of only three samples.
    for week in range(1, 18):
        rows.append(_prior_row('BUF', 'Mobile Starter', week, passing_attempts=32.0, rushing_attempts=5.0))
    for week in range(1, 18):
        rows.append(_prior_row('GB', 'Pocket Starter', week, passing_attempts=35.0, rushing_attempts=1.5))
    return pd.DataFrame(rows)


def test_thin_backup_sample_shrinks_hard_toward_league_and_team_anchor():
    history = _willis_like_prior_history()
    # cur frame has 2 QB rows for MIA: the selected QB1 (Willis) and a
    # non-starter who must stay untouched (NaN).
    identity_keys = np.array(['name:malikwillis', 'name:backup'])
    team_keys = np.array(['MIA', 'MIA'])
    qb1_mask = np.array([True, False])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0)

    # The non-starter must be untouched.
    assert np.isnan(rates['passing_attempts'][1])

    # Willis's raw personal rate: 9 att / 5 carries per game (36 dropbacks
    # total across 4 games). The blend must move his rush share MATERIALLY
    # toward the league mean rather than keeping his raw 5/(9+5)=0.357.
    raw_personal_rush_share = 5.0 / (9.0 + 5.0)
    assert audit['blended_rush_share'][0] < raw_personal_rush_share * 0.7
    # But it must still sit clearly above a typical starter's team rush
    # share (MIA's own prior-team QB ran 2/32 = 0.0625) - his own evidence
    # counts for something, it just cannot dominate a 4-game sample.
    assert audit['blended_rush_share'][0] > 2.0 / 32.0 * 1.3

    # Volume is anchored to the TEAM's own dropback capacity (MIA: 32/gm),
    # not to Willis's own tiny 14-dropback-per-game sample.
    assert _approx(audit['team_dropback_capacity'][0], 32.0, rel=0.05)
    total_volume = rates['passing_attempts'][0] + rates['rushing_attempts'][0]
    assert _approx(total_volume, 32.0, rel=0.05)
    # And the fix must move him meaningfully off today's raw 7.74-attempt
    # projection in the direction of a real starter's workload.
    assert rates['passing_attempts'][0] > 15.0


def test_established_starter_keeps_his_own_measured_style():
    history = _willis_like_prior_history()
    identity_keys = np.array(['name:establishedstarter'])
    team_keys = np.array(['KC'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0)

    # 17 games x 36 dropbacks = 612 dropbacks - evidence weight should
    # dominate the blend (612 / (612+200) ~= 0.75).
    assert audit['evidence_weight'][0] > 0.70
    raw_personal_rush_share = 2.0 / 36.0
    # With that much evidence, the blended share should land close to his
    # own measured rate, not near the (different, lower) league mean.
    assert _approx(audit['blended_rush_share'][0], raw_personal_rush_share, rel=0.35)


def test_thin_sample_efficiency_shrinks_toward_league_not_team():
    # Willis's fixture YPA is 7.0 (passing_yards = attempts * 7.0), same as
    # the "Established Starter"/"Prior Team QB" fixtures - so a shrunk blend
    # should land close to 7.0 regardless, which would not distinguish
    # "shrunk toward league" from "kept exactly as personal". Give Willis a
    # deliberately outlier personal YPA (12.0, matching the real 2025 small-
    # sample case: 422 yards / 35 attempts) so the two hypotheses diverge.
    history = _willis_like_prior_history()
    history.loc[history['player_display_name'].eq('Malik Willis'), 'passing_yards'] = (
        history.loc[history['player_display_name'].eq('Malik Willis'), 'passing_attempts'] * 12.0)
    identity_keys = np.array(['name:malikwillis'])
    team_keys = np.array(['MIA'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0)

    own_ypa = rates['passing_yards'][0] / rates['passing_attempts'][0]
    # Must move materially off the raw personal 12.0 YPA...
    assert own_ypa < 11.0
    # ...but 36 attempts of evidence (9/gm x 4 games) is not nothing, so it
    # should not collapse all the way to the league mean either.
    assert own_ypa > 7.5

    # A full-season starter's own well-supported YPA should barely move.
    identity_keys2 = np.array(['name:establishedstarter'])
    team_keys2 = np.array(['KC'])
    rates2, _ = qvb.blend_qb1_volume(identity_keys2, team_keys2, qb1_mask, history, k=200.0)
    own_ypa2 = rates2['passing_yards'][0] / rates2['passing_attempts'][0]
    assert _approx(own_ypa2, 7.0, rel=0.05)


def test_rushing_dependents_use_own_volume_not_passing_volume():
    history = _willis_like_prior_history()
    identity_keys = np.array(['name:malikwillis'])
    team_keys = np.array(['MIA'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0)

    own_rush_ypc = rates['rushing_yards'][0] / rates['rushing_attempts'][0]
    # Fixture rush YPC is 5.5 for every QB, so a correctly-shrunk blend
    # should land close to it regardless of evidence weight.
    assert _approx(own_rush_ypc, 5.5, rel=0.1)


def test_no_prior_history_is_a_no_op():
    identity_keys = np.array(['name:rookie'])
    team_keys = np.array(['MIA'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, pd.DataFrame(), k=200.0)

    assert np.isnan(rates['passing_attempts'][0])
    assert np.isnan(audit['blended_rush_share'][0])


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
