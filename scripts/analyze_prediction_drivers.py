"""
Enrich the shipped model's (DEFAULT_FEATURES) startable-pool player-week
outcomes with real, already-available context fields, and correlate the
prediction error against them - a data-driven hunt for what predicts a
dud/breakout week, rather than guessed-in-the-abstract factors.

Built 2026-08-27 per explicit request: analyze the backtest output, attach
age/team-projection/defense-projection/role plus 5+ more available fields,
find real correlations, and use those to propose new candidate model
components.

Every field joined here already exists somewhere in this codebase - nothing
new is fetched or estimated for this pass:

  - age / years_exp / is_rookie_flag       - roster bio (data.loaders, joined
                                              into load_and_merge_data's frame)
  - Role Confidence / Role Change Confidence / Expected Snap Share
                                            - the shipped model's OWN per-row
                                              output (data.weekly_projections)
  - _profile_target_share / _profile_snap_share / _profile_adot
                                            - the shipped model's own role
                                              profile columns
  - implied_points / implied_allowed / home / total_line
                                            - data.odds_market's historical
                                              Vegas lines (spread_line/
                                              total_line from nflverse's own
                                              game file - posted for PAST
                                              seasons, not a live-only feed)

IMPORTANT PRIOR ART - read before proposing anything Vegas-shaped:
data/odds_market.py's own TEAM_MULTIPLIER_MEASUREMENT already tested scaling
a projection by team implied points (748 player-seasons, 2023-2025) and found
it makes things WORSE pre-season and does ~nothing with hindsight lines.
data/weekly_projections.py's DEFAULT_FEATURES comment separately records
'game_env' (market total + roof/wind/rest, IN-SEASON weekly) as built,
measured, and rejected (+0.012 MAE). Any implied-points correlation this
script finds needs to clear THAT bar, not be re-proposed as if novel.

Usage:
    python scripts/analyze_prediction_drivers.py --years 2024,2025 --weeks 2-18
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.odds_market import fetch_game_lines, implied_team_points  # noqa: E402
from scripts.eval_weekly_model import STARTABLE_N, _actual_points  # noqa: E402

PROFILE_COL_MAP = {
    '_profile_target_share': 'target_share',
    '_profile_snap_share': 'snap_share_profile',
    '_profile_adot': 'adot',
    '_profile_carry_share': 'carry_share',
    '_profile_target_earner_score': 'target_earner_score',
}

NUMERIC_FEATURES = [
    'age', 'years_exp', 'is_rookie', 'role_confidence', 'role_change_confidence',
    'expected_snap_share', 'target_share', 'snap_share_profile', 'adot', 'carry_share',
    'target_earner_score', 'implied_points', 'implied_allowed', 'total_line', 'home',
]


def _scope_df(df, pos):
    d = df[df['Pos'] == pos]
    return d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')


def build_dataset(years, weeks, scoring='Full PPR'):
    records = []
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue

        # Season-level roster bio, one row per player (most recent snapshot).
        # 'age' does not survive load_and_merge_data's merge (confirmed empty
        # in this frame) - 'birth_date' does, so age is computed here as of
        # Sept 1 of the season (a stable, backtest-safe proxy for "age that
        # season" - no today-relative drift the way a live age would have).
        bio_cols = [c for c in ('birth_date', 'years_exp', 'is_rookie_flag') if c in stats_df.columns]
        bio = (stats_df[[name_col] + bio_cols].drop_duplicates(subset=[name_col], keep='last')
               .set_index(name_col)) if bio_cols else pd.DataFrame()
        if 'birth_date' in bio.columns:
            season_start = pd.Timestamp(year=year, month=9, day=1)
            bio['age'] = (season_start - pd.to_datetime(bio['birth_date'], errors='coerce')).dt.days / 365.25

        games, meta = fetch_game_lines(year)
        lines = implied_team_points(games) if not games.empty else pd.DataFrame()
        if not lines.empty:
            lines = lines.set_index(['week', 'team'])
        else:
            print(f"{year}: no Vegas lines available ({meta.get('error')})")

        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, 'fantasy_points_ppr')
            if actual.empty:
                continue
            proj, meta_p = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if proj.empty:
                continue

            for pos in ('QB', 'RB', 'WR', 'TE'):
                top = _scope_df(proj, pos)
                for _, r in top.iterrows():
                    player = r['Player']
                    if player not in actual.index:
                        continue
                    pred_v = float(r['Model Proj Pts'])
                    actual_v = float(actual[player])
                    rec = {
                        'year': year, 'week': week, 'player': player, 'pos': pos,
                        'team': r.get('Team'), 'pred': pred_v, 'actual': actual_v,
                        'signed_error': pred_v - actual_v, 'abs_error': abs(pred_v - actual_v),
                        'role_confidence': r.get('Role Confidence'),
                        'role_change_confidence': r.get('Role Change Confidence'),
                        'expected_snap_share': r.get('Expected Snap Share'),
                    }
                    for src, dst in PROFILE_COL_MAP.items():
                        if src in r.index:
                            rec[dst] = r[src]
                    if not bio.empty and player in bio.index:
                        row = bio.loc[player]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        rec['age'] = pd.to_numeric(row.get('age'), errors='coerce')
                        rec['years_exp'] = pd.to_numeric(row.get('years_exp'), errors='coerce')
                        rec['is_rookie'] = bool(row.get('is_rookie_flag', False))
                    if not lines.empty and (week, r.get('Team')) in lines.index:
                        lr = lines.loc[(week, r.get('Team'))]
                        if isinstance(lr, pd.DataFrame):
                            lr = lr.iloc[0]
                        rec['implied_points'] = lr.get('implied_points')
                        rec['implied_allowed'] = lr.get('implied_allowed')
                        rec['total_line'] = lr.get('total_line')
                        rec['home'] = lr.get('home')
                    records.append(rec)
    df = pd.DataFrame(records)
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def print_correlations(df):
    print(f"\n{'=' * 70}\nCOVERAGE (non-null / total rows)\n{'=' * 70}")
    for c in NUMERIC_FEATURES:
        n_ok = df[c].notna().sum()
        print(f"  {c:<24} {n_ok:>6} / {len(df)}  ({100 * n_ok / max(len(df), 1):.0f}%)")

    print(f"\n{'=' * 70}\nCORRELATION WITH |error| and signed error (pred - actual), BY POSITION\n"
          f"positive signed-error corr = feature associated with OVER-projection\n{'=' * 70}")
    for pos in ('QB', 'RB', 'WR', 'TE'):
        sub = df[df['pos'] == pos]
        print(f"\n  -- {pos} (n={len(sub)}) --")
        print(f"  {'feature':<24}{'r(|err|)':>10}{'n':>7}   {'r(signed)':>10}{'n':>7}")
        for c in NUMERIC_FEATURES:
            x = sub[c]
            ok_abs = x.notna() & sub['abs_error'].notna()
            ok_signed = x.notna() & sub['signed_error'].notna()
            if ok_abs.sum() < 20:
                continue
            r_abs = x[ok_abs].corr(sub.loc[ok_abs, 'abs_error'])
            r_signed = x[ok_signed].corr(sub.loc[ok_signed, 'signed_error'])
            print(f"  {c:<24}{r_abs:>10.3f}{int(ok_abs.sum()):>7}   {r_signed:>10.3f}{int(ok_signed.sum()):>7}")

    print(f"\n{'=' * 70}\nBUCKET COMPARISONS (mean |error|, all positions pooled unless noted)\n{'=' * 70}")
    rookie = df[df['is_rookie'] == True]['abs_error']  # noqa: E712
    vet = df[df['is_rookie'] == False]['abs_error']
    print(f"  rookie:        n={len(rookie):<6} mean|err|={rookie.mean():.3f}")
    print(f"  non-rookie:    n={len(vet):<6} mean|err|={vet.mean():.3f}")

    home = df[df['home'] == 1]['abs_error']
    away = df[df['home'] == 0]['abs_error']
    print(f"\n  home:          n={len(home):<6} mean|err|={home.mean():.3f}")
    print(f"  away:          n={len(away):<6} mean|err|={away.mean():.3f}")

    rc = df['role_change_confidence'].dropna()
    if not rc.empty:
        hi_thresh = rc.quantile(0.75)
        hi = df[df['role_change_confidence'] >= hi_thresh]['abs_error']
        lo = df[df['role_change_confidence'] < hi_thresh]['abs_error']
        print(f"\n  role_change_confidence >= {hi_thresh:.2f} (top quartile): n={len(hi):<6} mean|err|={hi.mean():.3f}")
        print(f"  role_change_confidence <  {hi_thresh:.2f}:                  n={len(lo):<6} mean|err|={lo.mean():.3f}")
        for pos in ('QB', 'RB', 'WR', 'TE'):
            sub = df[df['pos'] == pos]
            hi_p = sub[sub['role_change_confidence'] >= hi_thresh]['abs_error']
            lo_p = sub[sub['role_change_confidence'] < hi_thresh]['abs_error']
            if len(hi_p) >= 10 and len(lo_p) >= 10:
                print(f"    {pos}: hi n={len(hi_p):<5} mean={hi_p.mean():.3f}   "
                      f"lo n={len(lo_p):<5} mean={lo_p.mean():.3f}")

    rconf = df['role_confidence'].dropna()
    if not rconf.empty:
        lo_thresh = rconf.quantile(0.25)
        hi = df[df['role_confidence'] >= lo_thresh]['abs_error']
        lo = df[df['role_confidence'] < lo_thresh]['abs_error']
        print(f"\n  role_confidence <  {lo_thresh:.2f} (bottom quartile): n={len(lo):<6} mean|err|={lo.mean():.3f}")
        print(f"  role_confidence >= {lo_thresh:.2f}:                     n={len(hi):<6} mean|err|={hi.mean():.3f}")

    spread = (df['total_line']).dropna()
    if not spread.empty:
        hi_total = df['total_line'].quantile(0.75)
        lo_total = df['total_line'].quantile(0.25)
        hi = df[df['total_line'] >= hi_total]['abs_error']
        lo = df[df['total_line'] <= lo_total]['abs_error']
        print(f"\n  total_line >= {hi_total:.1f} (shootout games): n={len(hi):<6} mean|err|={hi.mean():.3f}")
        print(f"  total_line <= {lo_total:.1f} (low-scoring games): n={len(lo):<6} mean|err|={lo.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]}")
    df = build_dataset(years, weeks, args.scoring)
    print(f"collected {len(df)} startable player-weeks")
    print_correlations(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nfull enriched dataset written to {args.csv}")


if __name__ == '__main__':
    main()
