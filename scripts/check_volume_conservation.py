"""
Team-level volume conservation check for the weekly projection board.

WHY THIS EXISTS. The V2 audit (2026-08-22) found that passing and receiving
are projected in unreconciled silos: team targets summed to 1.19x-1.56x team
pass attempts depending on the week, entirely because WR/TE/RB target shares
are each drawn from a LEAGUE-WIDE position rate with no per-team budget check
(see docs/weekly_projection_model_v2_rebuild.md, "the headline defect"). A
star-player spot check never catches this - the top 3-6 pass catchers per
team are individually accurate; the excess is entirely in the tail (13th+
options on the team's own target board taking a real, nonzero share each).

This script makes that failure mode a five-second check instead of a
multi-hour audit: build a board, sum every relevant stat pair PER TEAM, and
report the ratio next to what real NFL teams actually do. A team-week is
"conserved" when targets/attempts, receptions/completions, and
receiving_yards/passing_yards all sit near 1.0 (targets/attempts is never
exactly 1.0 - throwaways, spikes, and batted balls at the line have no
targeted receiver - so the real-NFL reference column, not a bare 1.0, is the
actual bar).

Also reports Ourlads depth-chart team coverage (n/32) alongside the volume
numbers, since an incomplete source snapshot (see HANDOFF gotcha: DET/PIT
silently missing from a 30-team CSV) produces the same symptom as a real
model defect - a team with no depth-chart signal falls back to a flatter,
less differentiated role split - and the two are easy to conflate without
this printed side by side.

Usage:

    python scripts/check_volume_conservation.py --year 2026 --week 1
    python scripts/check_volume_conservation.py --year 2025 --week 10 --as-of-week 10 --model-version v2
    python scripts/check_volume_conservation.py --years 2026,2025,2025 --weeks 1,4,10 --model-version v2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 40)

# Stat pairs this check reconciles: (volume-side stat, opportunity-side stat,
# human label). Every one of these should sit near its real-NFL reference
# ratio at the TEAM level, regardless of which individual players earn the
# opportunity.
CONSERVATION_PAIRS = [
    ('targets', 'passing_attempts', 'targets / pass attempts'),
    ('receptions', 'passing_completions', 'receptions / completions'),
    ('receiving_yards', 'passing_yards', 'receiving yards / passing yards'),
    ('receiving_tds', 'passing_tds', 'receiving TDs / passing TDs'),
]

REFERENCE_STATS = [
    'passing_attempts', 'passing_yards', 'passing_tds', 'passing_completions',
    'targets', 'receptions', 'receiving_yards', 'receiving_tds',
    'rushing_attempts', 'rushing_yards',
]


def real_nfl_team_game_reference(year):
    """Real per-team-game stat means from that season's own box scores.

    A hardcoded reference number goes stale the moment league-wide pass
    rate shifts; deriving it from the same season under test keeps the bar
    honest even if a future season's passing environment looks nothing like
    2025's.
    """
    stats, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    if stats.empty or 'week' not in stats.columns:
        return {}
    frame = stats.copy()
    game_team_col = 'game_team' if 'game_team' in frame.columns else team_col
    frame['_team'] = frame[game_team_col].astype(str).str.upper()
    frame['_week'] = pd.to_numeric(frame['week'], errors='coerce')
    frame = frame[(frame['_team'] != '') & frame['_week'].notna()]
    if frame.empty:
        return {}
    for stat in REFERENCE_STATS:
        if stat not in frame.columns:
            frame[stat] = 0.0
        frame[stat] = pd.to_numeric(frame[stat], errors='coerce').fillna(0.0)
    team_week = frame.groupby(['_team', '_week'], observed=True)[REFERENCE_STATS].sum()
    per_team_game = team_week.groupby('_team', observed=True).mean()
    reference = per_team_game.mean().to_dict()
    reference['_teams'] = int(per_team_game.shape[0])
    reference['_team_games'] = int(team_week.shape[0])
    return reference


def ourlads_coverage_report():
    """32/32 or a named-gap warning, so a data gap is never read as a model bug."""
    try:
        from data.ourlads_depth_charts import load_ourlads_snapshot, TEAM_CONFIG
    except Exception as exc:
        return f'Ourlads coverage: could not check ({exc})'
    snapshot, err = load_ourlads_snapshot()
    if err:
        return f'Ourlads coverage: could not load snapshot ({err})'
    if snapshot is None or snapshot.empty:
        return 'Ourlads coverage: 0/32 - no snapshot imported'
    teams = sorted(snapshot['team'].dropna().astype(str).unique().tolist())
    missing = sorted(set(TEAM_CONFIG) - set(teams))
    status = f'Ourlads coverage: {len(teams)}/{len(TEAM_CONFIG)}'
    if missing:
        status += f' - MISSING: {", ".join(missing)}'
    return status


def team_conservation(result, year, meta=None):
    """Per-team ratios for every CONSERVATION_PAIRS entry, plus league summary.

    Teams with an unresolved QB1 room (source_contract's
    qb1_selection_required_teams - the model deliberately zeros EVERY QB's
    passing_attempts there until a user picks a starter, per its own
    documented "no guessing" policy) are EXCLUDED from the league-wide
    aggregate: dividing a real receiving estimate by a deliberate zero
    produces an infinite/meaningless ratio that is a harness artifact, not
    a volume-conservation defect. They are still reported by name so an
    unresolved room is visible rather than silently dropped.
    """
    if result is None or result.empty or 'Team' not in result.columns:
        return pd.DataFrame(), {}
    frame = result.copy()
    for stat in REFERENCE_STATS:
        if stat not in frame.columns:
            frame[stat] = 0.0
        frame[stat] = pd.to_numeric(frame[stat], errors='coerce').fillna(0.0)
    team_totals = frame.groupby('Team', observed=True)[REFERENCE_STATS].sum()
    reference = real_nfl_team_game_reference(year)
    out = pd.DataFrame(index=team_totals.index)
    for volume_stat, opp_stat, label in CONSERVATION_PAIRS:
        ratio = team_totals[volume_stat] / team_totals[opp_stat].replace(0, np.nan)
        out[label] = ratio
    unresolved = set((meta or {}).get('source_contract', {}).get('qb1_selection_required_teams', []) or [])
    conservable = team_totals.drop(index=[t for t in unresolved if t in team_totals.index])
    league_row = {}
    for volume_stat, opp_stat, label in CONSERVATION_PAIRS:
        denom = max(conservable[opp_stat].sum(), 1e-9)
        league_ratio = float(conservable[volume_stat].sum() / denom)
        real_ratio = (reference.get(volume_stat, np.nan) / reference.get(opp_stat, np.nan)
                     if reference else np.nan)
        league_row[label] = {'model': league_ratio, 'real_nfl': real_ratio,
                             'teams_over_1.0': int((out[label] > 1.0).sum())}
    league_row['_unresolved_qb1_teams'] = sorted(unresolved)
    return out, league_row


def run_one(year, week, as_of_week, apply_injury):
    result, meta = build_weekly_projections(
        year, week, as_of_week=as_of_week, apply_injury=apply_injury)
    if result is None or result.empty:
        print(f'  {year} week {week}: EMPTY BOARD - {meta.get("reason")}')
        return None
    per_team, league_row = team_conservation(result, year, meta)
    print(f'\n=== {year} week {week} (as-of {as_of_week}), '
         f'cold_start={meta.get("cold_start", "?")} ===')
    print(f'  players: {len(result)}, teams: {result["Team"].nunique()}')
    unresolved = league_row.pop('_unresolved_qb1_teams', [])
    if unresolved:
        print(f'  QB1 selection required (excluded from league ratio below): {", ".join(unresolved)}')
    for label, row in league_row.items():
        flag = ' <-- OVER 1.0 LEAGUE-WIDE' if row['model'] > 1.05 else ''
        real = f'{row["real_nfl"]:.3f}' if row['real_nfl'] == row['real_nfl'] else 'n/a'
        print(f'  {label:32s} model={row["model"]:.3f}  real_nfl={real}  '
             f'teams>1.0={row["teams_over_1.0"]}/32{flag}')
    worst = per_team.sort_values(per_team.columns[0], ascending=False).head(5)
    print('  worst 5 teams (targets/attempts):')
    print(worst.to_string(float_format=lambda x: f'{x:6.2f}'))
    return {'year': year, 'week': week,
           'league': league_row, 'per_team': per_team}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--week', type=int, default=None)
    parser.add_argument('--as-of-week', type=int, default=None)
    parser.add_argument('--years', type=str, default=None, help='comma-separated, paired with --weeks')
    parser.add_argument('--weeks', type=str, default=None, help='comma-separated, paired with --years')
    parser.add_argument('--apply-injury', action='store_true', default=False)
    args = parser.parse_args()

    print(ourlads_coverage_report())

    runs = []
    if args.years and args.weeks:
        years = [int(y) for y in args.years.split(',')]
        weeks = [int(w) for w in args.weeks.split(',')]
        if len(years) != len(weeks):
            raise SystemExit('--years and --weeks must have the same length')
        runs = list(zip(years, weeks))
    elif args.year and args.week:
        runs = [(args.year, args.week)]
    else:
        # Default sweep: reproduces the audit's three reference points.
        runs = [(2026, 1), (2025, 4), (2025, 10)]

    results = []
    for year, week in runs:
        as_of = args.as_of_week if args.as_of_week is not None else week
        results.append(run_one(year, week, as_of, args.apply_injury))
    return results


if __name__ == '__main__':
    main()
