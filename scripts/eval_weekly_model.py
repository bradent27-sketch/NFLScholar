"""
A/B harness for data.weekly_projections.build_weekly_projections.

WHY THIS EXISTS ALONGSIDE validate_weekly_projections.py. That script is the
project's ONE honest pass: build the model, compare it to a naive baseline,
report the numbers, don't tune against it. It answers "is the shipped model
any good". It cannot answer "does adding component X help", because it can
only run one configuration and reports unpaired aggregates.

This script answers the second question, and it is deliberately built so a
component can be REJECTED:

  - Every variant is evaluated on the SAME weeks and, per week, on the
    intersection of the player pools every variant produced. An unpaired
    comparison across variants that project slightly different player sets
    silently compares different populations, and a component that merely
    drops a few hard-to-project players would look like an improvement.
  - Deltas are reported per week as well as pooled, with the count of weeks
    each variant won. A component that improves the pooled average purely
    off one big week is not a component that works.
  - Two seasons by default. A single season's weeks 5-17 is ~4,000
    player-weeks, which sounds like a lot and is really ~13 correlated
    samples; the same change can move MAE by 0.05 in either direction
    between seasons (this has already happened to this model - see
    HISTORY_MATCHUP_CLIP's own comment).
  - A STARTABLE subset (top N per position by projection) is reported next
    to the full pool. The full pool is dominated by deep-bench players
    projected near zero who are trivially easy to rank, and a lineup
    decision is only ever made among the players anyone would actually
    start. A component can help the full pool and hurt the decisions.

Injuries are OFF in every run (apply_injury=False) for the reason
build_weekly_projections' own docstring gives: the injury feed carries no
historical week granularity, so backtesting with it on scores every week
against a designation from months later.

Usage:

    python scripts/eval_weekly_model.py --variants base,role_matchup
    python scripts/eval_weekly_model.py --years 2024,2025 --weeks 5-17 \
        --variants base,all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, MODEL_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402

# How many players per position anyone would actually consider starting in a
# 12-team league (1QB/2RB/3WR/1TE + flex + a bench worth of streamers). The
# "startable" metrics below are computed inside this cut of each variant's
# own projection, which is the population a start/sit call is made from.
STARTABLE_N = {'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20}


def _parse_variant(name):
    """'base' -> no features; 'all' -> every feature; otherwise a
    '+'-separated list of feature names."""
    if name in ('base', 'none'):
        return frozenset()
    if name == 'all':
        return frozenset(MODEL_FEATURES)
    feats = frozenset(p for p in name.split('+') if p)
    unknown = feats - set(MODEL_FEATURES)
    if unknown:
        raise SystemExit(f"unknown feature(s): {sorted(unknown)}; known: {sorted(MODEL_FEATURES)}")
    return feats


def _actual_points(stats_df, name_col, week, scoring_col):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    if scoring_col not in rows.columns:
        return pd.Series(dtype=float)
    return rows.groupby(name_col, observed=True)[scoring_col].sum()


def _naive_baseline(stats_df, name_col, as_of_week, scoring_col, n_weeks=4):
    hist = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') < as_of_week]
    if hist.empty or scoring_col not in hist.columns:
        return pd.Series(dtype=float)
    recent = hist.sort_values('week').groupby(name_col, observed=True).tail(n_weeks)
    return recent.groupby(name_col, observed=True)[scoring_col].mean()


def _metrics(pred, actual):
    joined = pd.concat([pred.rename('pred'), actual.rename('actual')], axis=1).dropna()
    if len(joined) < 5:
        return None
    err = joined['pred'] - joined['actual']
    return {
        'n': len(joined),
        'mae': float(err.abs().mean()),
        'rmse': float(np.sqrt((err ** 2).mean())),
        'bias': float(err.mean()),
        # Spearman via Pearson-on-ranks - no scipy in this project, and the
        # Pearson correlation of two rank series IS Spearman's rho.
        'rank_corr': float(joined['pred'].rank().corr(joined['actual'].rank())),
    }


def _weighted(rows, key):
    if not rows:
        return float('nan')
    n = np.array([r['n'] for r in rows], dtype=float)
    v = np.array([r[key] for r in rows], dtype=float)
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=n[ok])) if ok.any() else float('nan')


def evaluate(years, weeks, variants, scoring='Full PPR'):
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    # rows[variant][scope] = list of per-week metric dicts
    rows = {v: {} for v in variants}
    rows['_naive'] = {}

    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped")
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            projections = {}
            for vname, feats in variants.items():
                proj, meta = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
                if proj.empty:
                    print(f"{year} w{week} {vname}: nothing ({meta.get('reason')})")
                    projections = {}
                    break
                projections[vname] = proj
            if not projections:
                continue

            # PAIRED pool: only players every variant projected this week.
            pool = None
            for proj in projections.values():
                names = set(proj['Player'])
                pool = names if pool is None else (pool & names)
            pool = sorted(pool)
            if len(pool) < 20:
                continue

            for vname, proj in projections.items():
                sub = proj[proj['Player'].isin(pool)]
                pred = pd.Series(sub['Model Proj Pts'].to_numpy(), index=sub['Player'])
                m = _metrics(pred, actual)
                if m:
                    rows[vname].setdefault('ALL', []).append(m)
                for pos in ('QB', 'RB', 'WR', 'TE'):
                    pos_sub = sub[sub['Pos'] == pos]
                    pp = pd.Series(pos_sub['Model Proj Pts'].to_numpy(), index=pos_sub['Player'])
                    mp = _metrics(pp, actual)
                    if mp:
                        rows[vname].setdefault(pos, []).append(mp)
                    top = pos_sub.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
                    tp = pd.Series(top['Model Proj Pts'].to_numpy(), index=top['Player'])
                    mt = _metrics(tp, actual)
                    if mt:
                        rows[vname].setdefault(f'START-{pos}', []).append(mt)

            naive = _naive_baseline(stats_df, name_col, week, scoring_col).reindex(pool)
            mn = _metrics(naive, actual)
            if mn:
                rows['_naive'].setdefault('ALL', []).append(mn)
    return rows


def _print_scope(rows, scope, order):
    have = [v for v in order if rows.get(v, {}).get(scope)]
    if not have:
        return
    base = have[0]
    print(f"\n--- {scope} ---")
    header = f"{'variant':<26}{'n':>7}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'rankρ':>8}"
    if len(have) > 1:
        header += f"{'dMAE':>8}{'dρ':>8}{'wk W-L':>9}"
    print(header)
    base_rows = rows[base][scope]
    for v in have:
        rs = rows[v][scope]
        line = (f"{v:<26}{sum(r['n'] for r in rs):>7}{_weighted(rs, 'mae'):>8.3f}"
                f"{_weighted(rs, 'rmse'):>8.3f}{_weighted(rs, 'bias'):>+8.3f}"
                f"{_weighted(rs, 'rank_corr'):>8.3f}")
        if len(have) > 1 and v != base and len(rs) == len(base_rows):
            d_mae = _weighted(rs, 'mae') - _weighted(base_rows, 'mae')
            d_rho = _weighted(rs, 'rank_corr') - _weighted(base_rows, 'rank_corr')
            wins = sum(1 for a, b in zip(rs, base_rows) if a['mae'] < b['mae'])
            line += f"{d_mae:>+8.3f}{d_rho:>+8.3f}{f'{wins}-{len(rs) - wins}':>9}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='5-17')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--variants', default='base,all',
                    help="comma-separated; 'base', 'all', or feature names joined by '+'")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]
    variants = {name: _parse_variant(name) for name in args.variants.split(',')}

    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} scoring={args.scoring}")
    for name, feats in variants.items():
        print(f"  {name}: {sorted(feats) if feats else '(no optional components)'}")

    rows = evaluate(years, weeks, variants, args.scoring)
    order = list(variants) + ['_naive']
    for scope in ['ALL', 'QB', 'RB', 'WR', 'TE',
                  'START-QB', 'START-RB', 'START-WR', 'START-TE']:
        _print_scope(rows, scope, order)


if __name__ == '__main__':
    main()
