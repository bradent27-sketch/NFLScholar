"""
Fit the RB participation model behind 'v2_rb_participation'
(data/rb_participation.json).

The weekly model projects every RB at his rate WHEN ACTIVE and leaves "is he
playing at all" to the injury feed. For RB3 and below that question never
reaches the injury feed: healthy scratches and active bodies with no carries
are not injuries. Measured on 2024-2025 boards (clean rooms, nobody OUT): RB3
is projected 2.08 carries against 2.00 actual WHEN HE PLAYS, but only 58% of
RB3s have a box-score row (RB4+: 39%), so the room over-claims ~1.2 carries.
Projection x P(row) reproduces the actual total (1.21 vs 1.17).

P(row) is fit as a logistic on the boards as DEPLOYED (injured / IR players
already removed by the replay, so the injury feed's job is not double
counted), from:
    s      expected snap share entering the week (the trace's Expected Snap Share)
    app    games played this season / his team's games so far
    last1  appeared in his team's previous game
    app3   fraction of his team's previous 3 games he appeared in
Season-long appearance rate beats snap share by a wide margin (out of sample
log loss 0.336 vs 0.421; all four features 0.307). The factor is
g = min(1, P / ref) with ref = the actual appearance rate of the top group
(P >= 0.9, 0.966), so a healthy starter is ~1.0 and the residual miss rate
every player shares cancels; it is applied only to the 3rd-and-lower RB of a
room by expected snap share (RB1/RB2's conditional calibration is already
right and their P(row) >= 0.90).

Training boards come from scripts/diag_rb_carry_overprojection.py.

Usage:
    python scripts/diag_rb_carry_overprojection.py --years 2022,2023 --weeks 3-17 --out .sweeps/diag_rb_carries_2022-2023
    python scripts/fit_rb_participation.py --boards .sweeps/diag_rb_carries_2022-2023/rb_players.parquet
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'data', 'rb_participation.json')
FEATURES = ['s', 'app', 'last1', 'app3']
MIN_RANK = 3
REF_PROBABILITY = 0.9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(X, y, l2=1e-3, iters=60):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ b)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (y - p) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def recent_features(years):
    """{(year, name, week): (last1, app3, team_games_before)} for every RB
    from raw weekly stats - the same quantities the model computes live."""
    from data.transforms import load_and_merge_data
    out = {}
    for year in years:
        s, tc, nc, _ = load_and_merge_data(year, 'Full PPR')
        s = s.copy()
        s['w'] = pd.to_numeric(s['week'], errors='coerce')
        s['t'] = (s['game_team'] if 'game_team' in s.columns else s[tc]).astype(str).str.upper()
        s = s.dropna(subset=['w'])
        team_weeks = {t: np.array(sorted(set(x))) for t, x in s.groupby('t')['w']}
        rb = s[s['position'].astype(str) == 'RB']
        appear = {}
        for n, w, t in zip(rb[nc], rb['w'], rb['t']):
            appear.setdefault(n, {})[w] = t
        for name, m in appear.items():
            for w in range(3, 18):
                prev = [k for k in m if k < w]
                if not prev:
                    continue
                tw = team_weeks.get(m[max(prev)])
                if tw is None:
                    continue
                before = tw[tw < w]
                if len(before) == 0:
                    continue
                out[(year, name, w)] = (float(before[-1] in m), float(np.mean([k in m for k in before[-3:]])), len(before))
    return out


def prepare(boards_path):
    d = pd.read_parquet(boards_path)
    d = d[d['Availability'] > 0.01].copy()
    for c in ('rushing_attempts', 'expected_snap_share', 'current_games'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    rf = recent_features(sorted(d['year'].unique()))
    feats = [rf.get((y, n, w), (np.nan, np.nan, np.nan)) for y, n, w in zip(d['year'], d['Player'], d['week'])]
    d['last1'], d['app3'], d['team_games'] = zip(*feats)
    d['s'] = d['expected_snap_share'].fillna(0.0).clip(0, 1)
    d['app'] = (d['current_games'] / d['team_games']).clip(0, 1)
    d['appeared'] = d['played'].astype(float)
    d = d[(d['current_games'] >= 1) & d['last1'].notna()].copy()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', required=True, help='rb_players.parquet from diag_rb_carry_overprojection.py')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    d = prepare(args.boards)
    y = d['appeared'].to_numpy(float)
    b = fit_logit(d[FEATURES].to_numpy(float), y)
    p = sigmoid(np.column_stack([np.ones(len(d)), d[FEATURES].to_numpy(float)]) @ b)
    ref = float(d.loc[p >= REF_PROBABILITY, 'appeared'].mean())
    print(f"rows {len(d)}  P(row) {y.mean():.3f}  in-sample logloss {logloss(p, y):.4f}  (constant {logloss(np.full(len(y), y.mean()), y):.4f})")
    print("coef:", dict(zip(['intercept'] + FEATURES, np.round(b, 4))), " reference (P>=.9 group actual rate): %.4f" % ref)
    payload = {
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'boards': os.path.relpath(args.boards, ROOT).replace('\\', '/'),
        'years': sorted(int(v) for v in d['year'].unique()),
        'n': int(len(d)),
        'features': FEATURES,
        'intercept': round(float(b[0]), 5),
        'coefs': {f: round(float(v), 5) for f, v in zip(FEATURES, b[1:])},
        'ref': round(ref, 5),
        'min_rank': MIN_RANK,
    }
    if not args.no_write:
        with open(OUT_PATH, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"wrote {OUT_PATH}")


if __name__ == '__main__':
    main()
