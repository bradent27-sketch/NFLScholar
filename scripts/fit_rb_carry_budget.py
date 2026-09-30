"""
Fit the team RB carry budget behind 'v2_rb_carry_budget' (data/rb_carry_budget.json).

Uses data.rb_carry_budget's own feature builder, so the fit and the model share
one definition of every regressor. One row per team-game, weeks 3-17: features
from games BEFORE that week (shrunk toward last season), closing spread for the
game, target = the team's actual RB carries. OLS on (y - lg_rb).

Train years default to 2018-2021 so the 2022-2025 backtest window stays out of
sample; --oos reports the same model on later seasons.

Usage:
    python scripts/fit_rb_carry_budget.py
    python scripts/fit_rb_carry_budget.py --train 2018,2019,2020,2021 --oos 2022,2023,2024,2025
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.rb_carry_budget import (K_SHRINK, SPREAD_CLIP, _spread_by_team, budget_design,  # noqa: E402
                                  budget_features, team_game_frame)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'data', 'rb_carry_budget.json')


def season_rows(year, k, cache):
    from data.transforms import load_and_merge_data
    from data.loaders import load_schedule
    for y in (year - 1, year):
        if y not in cache:
            s, tc, _, _ = load_and_merge_data(y, 'Full PPR')
            cache[y] = team_game_frame(s, tc)
    games, prior = cache[year], cache[year - 1]
    sched = load_schedule(year)
    rows = []
    for week in range(3, 18):
        actual = games[games['week'] == week].set_index('team')['rb_carries']
        if actual.empty:
            continue
        f = budget_features(games[games['week'] < week], prior, _spread_by_team(sched, week), k=k)
        f = f[f.index.isin(actual.index)].copy()
        f['y'] = actual.reindex(f.index).to_numpy(float)
        f['year'], f['week'] = year, week
        rows.append(f)
    return pd.concat(rows) if rows else pd.DataFrame()


def build(years, k, cache):
    return pd.concat([season_rows(y, k, cache) for y in years])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='2018,2019,2020,2021')
    ap.add_argument('--oos', default='2022,2023,2024,2025')
    ap.add_argument('--k', type=float, default=K_SHRINK)
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    train_years = [int(v) for v in args.train.split(',')]
    oos_years = [int(v) for v in args.oos.split(',') if v]
    cache = {}
    tr = build(train_years, args.k, cache)
    X, y = budget_design(tr), (tr['y'] - tr['lg_rb']).to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = tr['lg_rb'].to_numpy() + X @ b
    print(f"train {train_years}: n={len(tr)}  mean RB carries {tr['y'].mean():.2f}  "
          f"in-sample RMSE {np.sqrt(((p - tr['y']) ** 2).mean()):.3f}")
    print("coef [c0, rate, spread(clip %.0f), opp_rb]:" % SPREAD_CLIP, np.round(b, 4))
    if oos_years:
        te = build(oos_years, args.k, cache)
        pt = te['lg_rb'].to_numpy() + budget_design(te) @ b
        naive = te['lg_rb'].to_numpy()
        for yr in oos_years:
            m = (te['year'] == yr).to_numpy()
            print(f"  OOS {yr}: n={m.sum()}  actual {te['y'][m].mean():.2f}  budget {pt[m].mean():.2f}  "
                  f"RMSE {np.sqrt(((pt[m] - te['y'][m]) ** 2).mean()):.3f}  (league mean {np.sqrt(((naive[m] - te['y'][m]) ** 2).mean()):.3f})"
                  f"  corr {np.corrcoef(pt[m], te['y'][m])[0, 1]:.3f}")
    payload = {
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'train_years': train_years, 'n': int(len(tr)), 'k_shrink': args.k, 'spread_clip': SPREAD_CLIP,
        'intercept': round(float(b[0]), 5),
        'coefs': {'rate': round(float(b[1]), 5), 'spread': round(float(b[2]), 5), 'opp_rb': round(float(b[3]), 5)},
    }
    if not args.no_write:
        with open(OUT_PATH, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
