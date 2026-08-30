"""
Mechanism check for the injury -> vacancy redistribution (NOT a leakage-free
backtest - the model deliberately disables historical injuries, so this
declares a player out "in hindsight" on purpose, per the user's own framing).

For a set of past (year, week) targets it:

  1. builds the normal historical board (v2_as_of_guard on, so no injuries),
  2. finds real IN-HINDSIGHT ABSENCES: a player who averaged a heavy snap
     share over the 3 weeks BEFORE `week` and then recorded no stat line at
     all in `week` (did not play),
  3. for each such player, zeroes his own board row (what `v2_availability`
     would have done) and runs the shipped redistribution
     (`redistribute_rb_vacancy_with_allocator` for core RBs, then
     `redistribute_v2_vacated_usage` for QB/WR/TE), exactly as
     `build_weekly_projections` does,
  4. reports the ledger AND whether the recipients' post-redistribution
     projections moved CLOSER to their real box score that week than the
     pre-redistribution projections were - the actual question ("does the
     distribution work").

Usage:
    python scripts/validate_injury_vacancy.py --year 2025 --weeks 6,9,12,15
    python scripts/validate_injury_vacancy.py --year 2025 --weeks 5-17 --pos RB
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import (  # noqa: E402
    build_weekly_projections, redistribute_v2_vacated_usage,
    redistribute_rb_vacancy_with_allocator,
)
from data.transforms import load_and_merge_data  # noqa: E402

SCORE_COL = 'fantasy_points_ppr'
VOLUME_BY_POS = {
    'RB': ['rushing_attempts', 'targets'],
    'WR': ['targets'], 'TE': ['targets'], 'QB': ['passing_attempts'],
}
PRE_SNAP_MIN = 55.0      # "was a real starter the weeks before"
PRE_GAMES_MIN = 2


def _actual_week(stats_df, name_col, week):
    wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    pts = wk.groupby(name_col, observed=True)[SCORE_COL].sum()
    played = set(wk.groupby(name_col, observed=True)['weekly_snap_pct'].max()
                 .pipe(lambda s: s[s.fillna(0) > 0]).index)
    return pts, played, wk


def _pre_window_starters(stats_df, name_col, week):
    pre = stats_df[(pd.to_numeric(stats_df['week'], errors='coerce') < week)
                   & (pd.to_numeric(stats_df['week'], errors='coerce') >= week - 3)]
    if pre.empty:
        return pd.DataFrame()
    g = pre.groupby([name_col, 'position'], observed=True).agg(
        snap=('weekly_snap_pct', 'mean'), games=('week', 'nunique'))
    g = g[(g['snap'] >= PRE_SNAP_MIN) & (g['games'] >= PRE_GAMES_MIN)].reset_index()
    return g


def _closer(before, after, actual):
    """+1 if `after` is closer to `actual` than `before`, -1 if worse, 0 tie."""
    b, a = abs(before - actual), abs(after - actual)
    return np.sign(b - a)


def run_week(year, week, pos_filter):
    stats_df, t_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    if 'week' not in stats_df.columns:
        print(f"{year} wk{week}: no weekly data"); return []
    actual_pts, played, wk_rows = _actual_week(stats_df, name_col, week)
    starters = _pre_window_starters(stats_df, name_col, week)
    if starters.empty:
        print(f"{year} wk{week}: no pre-window starters"); return []
    absent = starters[~starters[name_col].isin(played)]
    if pos_filter:
        absent = absent[absent['position'].str.upper().isin(pos_filter)]
    if absent.empty:
        print(f"{year} wk{week}: no in-hindsight absences among pre-window starters"); return []

    board, _meta = build_weekly_projections(year, week, 'Full PPR', as_of_week=week, apply_injury=False)
    if board.empty:
        print(f"{year} wk{week}: empty board"); return []
    board = board.copy()
    key = 'Player'

    records = []
    for _, row in absent.iterrows():
        player, ppos = row[name_col], str(row['position']).upper()
        if player not in set(board[key]):
            continue
        team = str(board.loc[board[key] == player, 'Team'].iloc[0])
        teammates = board[(board['Team'].astype(str) == team) & (board[key] != player)]

        before = board.set_index(key)
        one = board.copy()
        vol_cols = [c for c in VOLUME_BY_POS.get(ppos, []) if c in one.columns]
        dep = ['receptions', 'receiving_yards', 'receiving_tds', 'rushing_yards',
               'rushing_tds', 'passing_yards', 'passing_completions', 'passing_tds']
        one.loc[one[key] == player, [c for c in vol_cols + dep if c in one.columns]] = 0.0

        profiles = {player: {'plays_probability': 0.0, 'source_year': year,
                             'source': 'in-hindsight absence (validation)'}}
        if ppos == 'RB':
            one, rb_ledger = redistribute_rb_vacancy_with_allocator(
                one, profiles, as_of_year=year,
                injury_provenance={player: {'year': year, 'source': 'validation'}})
            one, _n, other_ledger = redistribute_v2_vacated_usage(
                one, profiles, skip_rb=True, skip_receivers=True)
            led = (rb_ledger.to_dict('records') if rb_ledger is not None and not rb_ledger.empty else []) + other_ledger
        else:
            one, _n, led = redistribute_v2_vacated_usage(one, profiles)

        after = one.set_index(key)
        moved = [e for e in led if float(e.get('allocated', 0) or 0) > 0]
        recips = set()
        for e in moved:
            for r in (e.get('recipients') or []):
                recips.add(r['player'])
        # Score: did the recipients' projected points move toward their actual?
        wins = losses = 0
        for r in recips:
            if r not in before.index or r not in after.index or r not in actual_pts.index:
                continue
            b = float(before.loc[r].get('Model Proj Pts', np.nan))
            a = float(after.loc[r].get('Model Proj Pts', np.nan))
            act = float(actual_pts.get(r, np.nan))
            if not np.isfinite(b) or not np.isfinite(a) or not np.isfinite(act):
                continue
            s = _closer(b, a, act)
            wins += s > 0
            losses += s < 0
        vac = sum(float(e.get('vacated', 0) or 0) for e in moved)
        alloc = sum(float(e.get('allocated', 0) or 0) for e in moved)
        records.append({
            'year': year, 'week': week, 'player': player, 'pos': ppos, 'team': team,
            'pre_snap': round(float(row['snap']), 1), 'vacated': round(vac, 1),
            'allocated': round(alloc, 1), 'unfilled': round(vac - alloc, 1),
            'n_recipients': len(recips), 'recip_proj_closer': wins,
            'recip_proj_worse': losses,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--weeks', default='6,9,12,15')
    ap.add_argument('--pos', default='', help="comma list e.g. RB,WR - default all")
    args = ap.parse_args()
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-'); weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    pos_filter = [p.strip().upper() for p in args.pos.split(',') if p.strip()]

    all_rows = []
    for w in weeks:
        all_rows += run_week(args.year, w, pos_filter)
    if not all_rows:
        print("no cases found"); return
    df = pd.DataFrame(all_rows)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
    print("\n=== per-case ===")
    print(df.to_string(index=False))
    print("\n=== summary ===")
    print(f"cases: {len(df)}")
    print(f"mean vacated: {df['vacated'].mean():.1f}  mean allocated: {df['allocated'].mean():.1f}  "
          f"mean unfilled: {df['unfilled'].mean():.1f}  "
          f"({100 * df['allocated'].sum() / max(df['vacated'].sum(), 1e-9):.0f}% of vacated volume re-placed)")
    tot_closer, tot_worse = df['recip_proj_closer'].sum(), df['recip_proj_worse'].sum()
    print(f"recipient projections vs their actual box score: {tot_closer} moved CLOSER, "
          f"{tot_worse} moved WORSE  "
          f"({100 * tot_closer / max(tot_closer + tot_worse, 1):.0f}% closer)")
    by_pos = df.groupby('pos').agg(
        cases=('player', 'size'), pct_placed=('allocated', 'sum'),
        vac=('vacated', 'sum'), closer=('recip_proj_closer', 'sum'),
        worse=('recip_proj_worse', 'sum'))
    by_pos['pct_placed'] = (100 * by_pos['pct_placed'] / by_pos['vac']).round(0)
    print("\nby position:")
    print(by_pos[['cases', 'pct_placed', 'closer', 'worse']].to_string())


if __name__ == '__main__':
    main()
