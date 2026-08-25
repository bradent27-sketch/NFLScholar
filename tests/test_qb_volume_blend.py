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


def test_prior_history_team_capacity_uses_raw_history_not_personal_eligible_filter():
    # Personal frame: only the QB's 3 full-role starts (weeks 1-3) - this is
    # what a caller passes after filtering out a QB-split/relief game via
    # annotate_player_history_participation's _player_history_eligible flag
    # (see weekly_projections.py's player_prior, and the real Jayden Daniels
    # 2025 case that exposed this: two QB-split relief games were dragging
    # his own personal rate down even though the exclusion was already
    # computed and displayed - just never applied). Team frame: the RAW,
    # unfiltered history, which ALSO has a week-4 split game (a 10-attempt
    # relief stint from the QB1 plus a 15-attempt relief stint from a
    # backup) - team dropback capacity must still see that week's real
    # total, even though neither individual row belongs in a personal rate
    # (same "raw team-game history" principle the defense side already
    # relies on).
    eligible_history = pd.DataFrame([
        _prior_row('WAS', 'Star QB', 1, 30, 5),
        _prior_row('WAS', 'Star QB', 2, 30, 5),
        _prior_row('WAS', 'Star QB', 3, 30, 5),
    ])
    raw_history = pd.concat([eligible_history, pd.DataFrame([
        _prior_row('WAS', 'Star QB', 4, 10, 2),
        _prior_row('WAS', 'Backup QB', 4, 15, 3),
    ])], ignore_index=True)

    identity_keys = np.array(['name:starqb'])
    team_keys = np.array(['WAS'])
    qb1_mask = np.array([True])

    rates_split, audit_split = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, eligible_history, prior_history_team=raw_history)
    rates_same, audit_same = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, eligible_history)

    # Personal dropbacks: identical either way - always read off the
    # eligible-only frame passed as prior_history; week 4 never enters it
    # regardless of what prior_history_team carries.
    assert _approx(audit_split['personal_dropbacks'][0], 105.0)
    assert _approx(audit_same['personal_dropbacks'][0], 105.0)

    # Team capacity DIFFERS. With the raw team frame, week 4's real 30-
    # dropback team total (10+2 from the QB1's relief stint, 15+3 from the
    # backup's) is included: (35+35+35+30)/4 = 33.75. Omitting
    # prior_history_team falls back to the eligible-only frame for team
    # capacity too (backward compatible with every existing caller/test
    # that only ever had one frame to give) - week 4 never appears: 35.0.
    assert _approx(audit_split['team_dropback_capacity'][0], 33.75)
    assert _approx(audit_same['team_dropback_capacity'][0], 35.0)
    assert rates_split['passing_attempts'][0] != rates_same['passing_attempts'][0]


def _prior2_row(team, player, week, passing_attempts, rushing_attempts, **kwargs):
    return _prior_row(team, player, week, passing_attempts, rushing_attempts, **kwargs)


def test_prior2_blend_lifts_a_thin_injury_shortened_2025_toward_a_stronger_2024():
    # Lamar Jackson/Jayden Daniels shape: a down/injury-shortened 2025 (4
    # games) that understates a QB's real rushing style/efficiency relative
    # to a full, stronger 2024. history2025 gives him a modest rush share
    # (3/33 per game) and average YPA (6.0); history2024 gives him a
    # materially higher rush share (8/40) and YPA (8.0) over a full season.
    # NYJ is unused by _willis_like_prior_history()'s own fixtures, so this
    # QB's team-week rows don't double up with another QB's on the same
    # team/week when concatenated onto it.
    history2025 = pd.concat([_willis_like_prior_history(), pd.DataFrame([
        _prior_row('NYJ', 'Comeback QB', week, passing_attempts=30.0, rushing_attempts=3.0,
                   passing_yards=30.0 * 6.0)
        for week in (2, 6, 11, 15)
    ])], ignore_index=True)
    history2024 = pd.DataFrame([
        _prior2_row('NYJ', 'Comeback QB', week, passing_attempts=32.0, rushing_attempts=8.0,
                    passing_yards=32.0 * 8.0)
        for week in range(1, 18)
    ])
    identity_keys = np.array(['name:comebackqb'])
    team_keys = np.array(['NYJ'])
    qb1_mask = np.array([True])

    baseline_rates, baseline_audit = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, history2025, k=200.0)
    lifted_rates, lifted_audit = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, history2025, k=200.0, prior2_history=history2024)

    assert lifted_audit['prior2_weight'][0] > 0.30  # thin 2025 sample -> well above the 0.20 floor
    # 2024 raises both style (rush share) and efficiency (YPA) - both must
    # move UP relative to the 2025-only baseline, not just get shrunk.
    assert lifted_audit['blended_rush_share'][0] > baseline_audit['blended_rush_share'][0]
    baseline_ypa = baseline_rates['passing_yards'][0] / baseline_rates['passing_attempts'][0]
    lifted_ypa = lifted_rates['passing_yards'][0] / lifted_rates['passing_attempts'][0]
    assert lifted_ypa > baseline_ypa


def test_prior2_blend_dampens_a_2025_breakout_off_a_thin_2024_role():
    # A full, established 2025 season (17 games, floor weight) with a HIGH
    # rush share, against a 2024 where the same player ran a much lower rush
    # share (a pocket-passer role the year before the breakout). The blend
    # must still pull toward 2024, but only at PRIOR2_BLEND_DECREASE_
    # DAMPENING of the base weight - not the full floor weight a symmetric
    # blend would apply.
    history2025 = pd.DataFrame([
        _prior_row('MIA', 'Breakout QB', week, passing_attempts=25.0, rushing_attempts=11.0)
        for week in range(1, 18)
    ])
    history2024 = pd.DataFrame([
        _prior2_row('MIA', 'Breakout QB', week, passing_attempts=32.0, rushing_attempts=2.0)
        for week in range(1, 18)
    ])
    identity_keys = np.array(['name:breakoutqb'])
    team_keys = np.array(['MIA'])
    qb1_mask = np.array([True])

    baseline_rates, baseline_audit = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, history2025, k=200.0)
    dampened_rates, dampened_audit = qvb.blend_qb1_volume(
        identity_keys, team_keys, qb1_mask, history2025, k=200.0, prior2_history=history2024)

    assert dampened_audit['prior2_weight'][0] > 0.0
    actual_drop = baseline_audit['personal_rush_share'][0] - dampened_audit['personal_rush_share'][0]
    raw_2025_share = 11.0 / 36.0
    raw_2024_share = 2.0 / 34.0
    naive_undamped_drop = qvb.QB_PRIOR2_BASE_WEIGHT * (raw_2025_share - raw_2024_share)
    # The dampened drop must be well under half of what an undamped 20%
    # blend toward the same 2024 value would have produced.
    assert 0 < actual_drop < naive_undamped_drop * 0.6


def test_prior2_blend_is_a_no_op_without_2024_history():
    history = _willis_like_prior_history()
    identity_keys = np.array(['name:establishedstarter'])
    team_keys = np.array(['KC'])
    qb1_mask = np.array([True])

    _, audit_none = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0,
                                         prior2_history=None)
    _, audit_empty = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history, k=200.0,
                                          prior2_history=pd.DataFrame())
    assert audit_none['prior2_weight'][0] == 0.0
    assert audit_empty['prior2_weight'][0] == 0.0


def test_prior2_component_weight_zero_dropbacks_or_thin_games_is_a_no_op():
    assert qvb._prior2_component_weight(games_2025=16.0, games_2024=17.0, dropbacks_2024=0.0) == 0.0
    assert qvb._prior2_component_weight(games_2025=16.0, games_2024=0.0, dropbacks_2024=500.0) == 0.0
    assert qvb._prior2_component_weight(games_2025=16.0, games_2024=17.0, dropbacks_2024=500.0) > 0.0


def test_blend_component_toward_prior2_is_asymmetric():
    raised = qvb._blend_component_toward_prior2(current=0.10, prior2_value=0.30, weight_base=0.20)
    lowered = qvb._blend_component_toward_prior2(current=0.30, prior2_value=0.10, weight_base=0.20)
    assert _approx(raised, 0.10 + 0.20 * (0.30 - 0.10))
    assert _approx(lowered, 0.30 - (0.20 * qvb.QB_PRIOR2_DECREASE_DAMPENING) * (0.30 - 0.10))
    # Raising the value moves it further (full weight) than lowering it
    # (dampened weight) for the same-size gap.
    assert (raised - 0.10) > (0.30 - lowered)


def _full_season_mobile_qb_history():
    """One full-season mobile starter with a real rushing sample, plus an
    ordinary league backdrop, to exercise the DEFAULT (unspecified k) blend.

    Modeled on the 2026-08-25 user complaint: a full-season rusher's own
    measured rush share/YPC should barely be diluted toward the league mean,
    since QB rushing is almost entirely an individual trait, not a team-
    scheme effect.
    """
    rows = []
    for week in range(1, 18):
        # 34 attempts + 8 carries/gm, well above the league mean rush share,
        # with a real per-carry average distinct from the league mean too.
        rows.append(_prior_row('BAL', 'Mobile Starter', week, passing_attempts=34.0,
                               rushing_attempts=8.0, rushing_yards=8.0 * 6.0))
    for week in range(1, 18):
        rows.append(_prior_row('GB', 'Pocket Starter', week, passing_attempts=35.0, rushing_attempts=1.5))
    for week in range(1, 18):
        rows.append(_prior_row('KC', 'Established Starter', week, passing_attempts=34.0, rushing_attempts=2.0))
    return pd.DataFrame(rows)


def test_default_k_barely_dilutes_a_full_season_rushers_own_share_and_ypc():
    history = _full_season_mobile_qb_history()
    identity_keys = np.array(['name:mobilestarter'])
    team_keys = np.array(['BAL'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history)

    raw_rush_share = 8.0 / 42.0
    # The default (lowered 2026-08-25) K must leave a full season of
    # evidence overwhelmingly self-weighted - materially tighter than the
    # old K=200 behavior, which topped out around 75-86% self-weight here.
    assert audit['evidence_weight'][0] > 0.90
    assert _approx(audit['blended_rush_share'][0], raw_rush_share, rel=0.10)

    own_rush_ypc = rates['rushing_yards'][0] / rates['rushing_attempts'][0]
    # Personal YPC is 6.0; must land close to it, not pulled hard toward a
    # league mean YPC that differs from it.
    assert _approx(own_rush_ypc, 6.0, rel=0.10)


def test_default_k_still_shrinks_a_genuinely_thin_backup_sample():
    history = _willis_like_prior_history()
    identity_keys = np.array(['name:malikwillis'])
    team_keys = np.array(['MIA'])
    qb1_mask = np.array([True])

    rates, audit = qvb.blend_qb1_volume(identity_keys, team_keys, qb1_mask, history)

    raw_personal_rush_share = 5.0 / (9.0 + 5.0)
    # The Malik-Willis correction this module was built for must survive the
    # 2026-08-25 K changes (200 -> 30 -> 10): a genuine 4-game relief sample
    # still gets SOME real correction off its raw rate, just a much lighter
    # touch now that the user has asked twice to trust a QB's own raw
    # rushing numbers more - it is no longer the primary defense against a
    # small-sample outlier, only a softener of the most extreme cases.
    assert audit['blended_rush_share'][0] < raw_personal_rush_share * 0.95
    assert audit['evidence_weight'][0] < 0.90


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
