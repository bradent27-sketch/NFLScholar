"""
Fit the WR/TE participation curve behind 'v2_wrte_participation'
(data/participation_curve.json).

The weekly model projects every receiver at his share-WHEN-ACTIVE
(expected_snap_share's docstring: "is he playing at all" is left to the
injury feed). For a depth receiver that question never reaches the injury
feed - a healthy scratch, or an active body who never sees an offensive
snap, isn't an injury - so each one projects as if he'll play, the room's
claim runs over budget, and the uniform pass-capacity trim then takes that
surplus out of the starters too. scripts/diag_target_flatness.py measured
it (2022-2025 wk3-17): share slope 1.09, ranks 6-8 in no box score 44% of
the time, ranks 9+ 66%.

This curve is P(player has a box-score row in week w | his expected active
snap share s entering week w), relative to an established starter (s >= TOP),
so the injury-miss rate every player shares - which the live injury feed
already handles - cancels out and only the depth-specific part remains.
Built from raw weekly stats (no board builds), exactly the way
expected_snap_share computes s: mean snap share over his last 4
appearances before week w.

Usage:
    python scripts/fit_participation_curve.py --years 2019,2020,2021 --weeks 3-17
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'participation_curve.json')
TOP = 0.70
EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0]


def player_weeks(years, weeks):
    from data.transforms import load_and_merge_data
    rows = []
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df[stats_df['position'].astype(str).isin(['WR', 'TE'])].copy()
        s['week'] = pd.to_numeric(s['week'], errors='coerce')
        s['snap'] = (pd.to_numeric(s['weekly_snap_pct'], errors='coerce') / 100.0).clip(0, 1)
        s['team'] = (s['game_team'] if 'game_team' in s.columns else s[team_col]).astype(str).str.upper()
        s = s.dropna(subset=['week'])
        all_rows = stats_df.copy()
        all_rows['week'] = pd.to_numeric(all_rows['week'], errors='coerce')
        all_rows['team'] = (all_rows['game_team'] if 'game_team' in all_rows.columns
                            else all_rows[team_col]).astype(str).str.upper()
        team_weeks = set(zip(all_rows['team'], all_rows['week']))
        appeared = set(zip(s[name_col], s['week']))
        for w in weeks:
            hist = s[(s['week'] < w)].dropna(subset=['snap']).sort_values('week')
            if hist.empty:
                continue
            recent = hist.groupby(name_col).tail(4)
            share = recent.groupby(name_col)['snap'].mean()
            last = hist.groupby(name_col).agg(team=('team', 'last'), pos=('position', 'last'))
            for player, sh in share.items():
                team = last.at[player, 'team']
                if (team, w) not in team_weeks:
                    continue  # bye (or team gone) - nothing to appear in
                rows.append({'year': year, 'week': w, 'player': player, 'pos': last.at[player, 'pos'],
                             's': float(sh), 'appeared': (player, w) in appeared})
    return pd.DataFrame(rows)


def _isotonic(values, weights):
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence.
    A bigger active role can't make a player LESS likely to be out there."""
    blocks = [[float(v), float(w), 1] for v, w in zip(values, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, w1, c1 = blocks[i]
            v2, w2, c2 = blocks.pop(i + 1)
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2]
            i = max(i - 1, 0)
        else:
            i += 1
    return [v for v, _w, c in blocks for _ in range(c)]


def fit(df):
    top = float(df.loc[df['s'] >= TOP, 'appeared'].mean())
    df = df.assign(bin=pd.cut(df['s'], EDGES, include_lowest=True))
    g = df.groupby('bin', observed=True).agg(n=('appeared', 'size'), p=('appeared', 'mean'),
                                             s_mean=('s', 'mean'))
    g['relative'] = (g['p'] / top).clip(upper=1.0)
    g['smoothed'] = np.minimum(_isotonic(g['relative'], g['n']), 1.0)
    return top, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2019,2020,2021')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(',')]
    lo, hi = (int(x) for x in args.weeks.split('-'))
    df = player_weeks(years, list(range(lo, hi + 1)))
    top, g = fit(df)
    print(f"player-weeks n={len(df)}  P(appear | s >= {TOP}) = {top:.3f}")
    print(g.round(3).to_string())
    for pos in ('WR', 'TE'):
        t, gp = fit(df[df['pos'] == pos])
        print(f"\n{pos}: top {t:.3f}")
        print(gp[['n', 'p', 'relative']].round(3).to_string())
    knots = [[round(float(r.s_mean), 4), round(float(r.smoothed), 4)]
             for r in g.itertuples() if r.s_mean < TOP]
    # Anchor the top of the curve at exactly 1.0 so an established starter is
    # never discounted by interpolation noise between the last two bins.
    knots.append([TOP, 1.0])
    knots = sorted({k[0]: k for k in knots}.values())
    payload = {'fit_years': years, 'weeks': f'{lo}-{hi}', 'positions': ['WR', 'TE'],
               'top_share': TOP, 'p_appear_top': round(top, 4), 'n': int(len(df)),
               'knots': knots}
    if not args.no_write:
        with open(OUT_PATH, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {OUT_PATH}")
    print(json.dumps(payload['knots']))


if __name__ == '__main__':
    main()
