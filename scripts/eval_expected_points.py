"""
Do expected fantasy points (nflverse `ff_opportunity`: expected receptions / yards / TDs from each touch's context) carry
information the weekly model does not already use? (2026-10-19, backlog item N1.)

The model's rates are built from what players DID. A player whose results ran ahead of his usage (actual minus expected
points, "luck") should regress; one behind it should rebound. Test:

    luck_pg = mean over the player's EARLIER games this season of (actual PPR points - expected PPR points)

1. Does luck_pg predict the shipped board's miss (actual - shown) in the target week? Slope by position, bootstrap CI over
   weeks, per-season sign. Negative = lucky players are over-projected.
2. A candidate, held out leave-one-season-out: shown' = shown + b * luck_pg * g/(g + K), b and K fit on the other seasons,
   scored like harness v2 (paired START pools, played rows, bootstrap CI over weeks) against the shipped board.
3. The same with the player's PRIOR season's luck, and for the stat that carries it (TDs vs yards vs catches).

The shipped board here is the lab dump (the harness base arm) with the shipped late steps re-applied
(`scripts.diag_returning_players.shipped_board`). The expected-stat model is nflverse's current one, trained on all seasons,
so its values for a past game are in-sample for the expectation model (not for the player's later results) - a mild
optimism to keep in mind; weeks of 2026 are the clean check once enough have been played.

    python scripts/eval_expected_points.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.utils import clean_name_exact  # noqa: E402
from scripts.diag_returning_players import shipped_board  # noqa: E402
from scripts.eval_calibration_schemes import boot_ci, load_dump, paired_scores  # noqa: E402

CACHE = os.path.join('.sweeps', 'ff_opportunity_weekly.parquet')
SEASONS = tuple(range(2020, 2027))
POSITIONS = ('RB', 'WR', 'TE', 'QB')


def load_ff(seasons=SEASONS, refresh=False):
    """Weekly ff_opportunity rows for ``seasons`` (cached in .sweeps), with the columns the evaluator reads."""
    if os.path.exists(CACHE) and not refresh:
        return pd.read_parquet(CACHE)
    import nflreadpy
    df = nflreadpy.load_ff_opportunity(seasons=list(seasons), stat_type='weekly').to_pandas()
    keep = ['season', 'week', 'posteam', 'player_id', 'full_name', 'position', 'rec_attempt', 'rush_attempt', 'pass_attempt',
            'receptions', 'receptions_exp', 'rec_yards_gained', 'rec_yards_gained_exp', 'rec_touchdown', 'rec_touchdown_exp',
            'rush_yards_gained', 'rush_yards_gained_exp', 'rush_touchdown', 'rush_touchdown_exp',
            'pass_yards_gained', 'pass_yards_gained_exp', 'pass_touchdown', 'pass_touchdown_exp',
            'total_fantasy_points', 'total_fantasy_points_exp', 'rec_fantasy_points', 'rec_fantasy_points_exp',
            'rush_fantasy_points', 'rush_fantasy_points_exp', 'pass_fantasy_points', 'pass_fantasy_points_exp']
    df = df[[c for c in keep if c in df.columns]].copy()
    os.makedirs('.sweeps', exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def luck_features(ff):
    """Per (key, season, week): how many earlier games, mean luck per earlier game this season, and the same split into
    touchdown luck (TDs above expected x 6 pts) and the rest. Also the prior season's mean luck per game."""
    f = ff.copy()
    f['key'] = clean_name_exact(f['full_name'].astype(str))
    f['season'] = pd.to_numeric(f['season'], errors='coerce').astype('int64')
    f['week'] = pd.to_numeric(f['week'], errors='coerce')
    f = f[f.week.between(1, 18)]
    f['luck'] = f['total_fantasy_points'] - f['total_fantasy_points_exp']
    td_act = f[['rec_touchdown', 'rush_touchdown', 'pass_touchdown']].fillna(0)
    td_exp = f[['rec_touchdown_exp', 'rush_touchdown_exp', 'pass_touchdown_exp']].fillna(0)
    td_pts = np.array([6.0, 6.0, 4.0])
    f['td_luck'] = ((td_act.to_numpy() - td_exp.to_numpy()) * td_pts).sum(axis=1)
    f['other_luck'] = f['luck'] - f['td_luck']
    f = f.sort_values(['key', 'season', 'week'])
    g = f.groupby(['key', 'season'])
    for c in ('luck', 'td_luck', 'other_luck', 'total_fantasy_points', 'total_fantasy_points_exp'):
        f['cum_' + c] = g[c].cumsum() - f[c]                  # sum over EARLIER games only
    f['games_before_ff'] = g.cumcount()
    prior = (f.groupby(['key', 'season']).agg(p_luck=('luck', 'sum'), p_games=('luck', 'size'),
                                              p_xfp=('total_fantasy_points_exp', 'sum')).reset_index())
    prior['season'] += 1
    f = f.merge(prior, on=['key', 'season'], how='left')
    f['luck_pg'] = f['cum_luck'] / f['games_before_ff'].where(f['games_before_ff'] > 0)
    f['td_luck_pg'] = f['cum_td_luck'] / f['games_before_ff'].where(f['games_before_ff'] > 0)
    f['other_luck_pg'] = f['cum_other_luck'] / f['games_before_ff'].where(f['games_before_ff'] > 0)
    f['xfp_pg'] = f['cum_total_fantasy_points_exp'] / f['games_before_ff'].where(f['games_before_ff'] > 0)
    f['prior_luck_pg'] = f['p_luck'] / f['p_games'].where(f['p_games'] >= 6)
    return f[['key', 'season', 'week', 'games_before_ff', 'luck_pg', 'td_luck_pg', 'other_luck_pg', 'xfp_pg', 'prior_luck_pg']]


def attach(d, feats):
    d = d.copy()
    d['key'] = clean_name_exact(d['player'].astype(str))
    # a name that appears twice in one week of the expected-stat file is ambiguous: no features rather than a guess
    feats = feats.drop_duplicates(['key', 'season', 'week'], keep=False)
    out = d.merge(feats.rename(columns={'season': 'year'}), on=['key', 'year', 'week'], how='left')
    assert len(out) == len(d)
    out.index = d.index
    return out


def slope_table(d, mask, cols=('luck_pg', 'td_luck_pg', 'other_luck_pg', 'prior_luck_pg')):
    rows = []
    for pos in POSITIONS:
        for col in cols:
            s = d[mask & (d['pos'] == pos) & d[col].notna()]
            if len(s) < 100:
                continue
            x, y = s[col].to_numpy(float), (s['actual'] - s['shown']).to_numpy(float)
            b = np.polyfit(x, y, 1)
            by_year = [np.polyfit(g[col], g['actual'] - g['shown'], 1)[0] for _, g in s.groupby('year') if len(g) > 60]
            # bootstrap over (year, week)
            wk = s['year'] * 100 + s['week']
            groups = {k: idx for k, idx in s.groupby(wk).indices.items()}
            keys = list(groups)
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(400):
                pick = rng.choice(len(keys), len(keys))
                ix = np.concatenate([groups[keys[i]] for i in pick])
                boots.append(np.polyfit(x[ix], y[ix], 1)[0])
            rows.append({'pos': pos, 'feature': col, 'n': len(s), 'slope': b[0], 'ci': f'[{np.percentile(boots, 2.5):+.2f},{np.percentile(boots, 97.5):+.2f}]',
                         'seasons negative': f'{sum(v < 0 for v in by_year)}/{len(by_year)}', 'sd of feature': x.std()})
    return pd.DataFrame(rows)


def loyo_luck(d, positions, k_games, feature='luck_pg', scale=1.0):
    """Held-out luck correction: shown + b[pos] * luck * g/(g+K), b fit by least squares through the origin on the OTHER
    seasons' played rows (weeks 3-17). Returns (predictions, {year: {pos: b}})."""
    pred = d['final'].to_numpy(float).copy()
    luck = d[feature].to_numpy(float)
    g = d['games_before_ff'].to_numpy(float)
    shrink = np.where(np.isnan(g), 0.0, g / (g + k_games)) if k_games > 0 else np.where(np.isnan(g), 0.0, 1.0)
    x = np.nan_to_num(luck, nan=0.0) * shrink
    played = (d['played'] & d['actual'].notna() & d['week'].between(3, 17)).to_numpy() & ~np.isnan(luck)
    live = (d['avail'] > 0.01).to_numpy() & (pred > 0.005)
    resid = (d['actual'] - d['final']).to_numpy(float)
    params = {}
    for y in sorted(d['year'].unique()):
        fit = played & (d['year'] != y).to_numpy()
        tgt = (d['year'] == y).to_numpy() & live
        params[y] = {}
        for pos in positions:
            pm = (d['pos'] == pos).to_numpy()
            f = fit & pm
            if f.sum() < 200:
                continue
            b = float((x[f] * resid[f]).sum() / (x[f] ** 2).sum())
            params[y][pos] = b
            pred[tgt & pm] = np.clip(pred[tgt & pm] + scale * b * x[tgt & pm], 0.0, None)
    return pred, params


def report(d, variants, window):
    base = d['final'].to_numpy(float)
    rows = []
    for name, (pred, params) in variants.items():
        sc = paired_scores(d, base, pred, window)
        r = {'variant': name}
        for scope in ('START-ALL', 'START-QB', 'START-RB', 'START-WR', 'START-TE', 'ALL'):
            if scope not in sc:
                continue
            x = sc[scope]
            dr = x['rmse_v'] - x['rmse_b']
            if scope in ('START-ALL', 'ALL'):
                lo, hi = boot_ci(x['weeks'])
                r[scope] = f'{dr:+.4f} [{lo:+.3f},{hi:+.3f}]'
            else:
                r[scope] = f'{dr:+.4f}'
        sa = sc['START-ALL']
        r['bias'] = f"{sa['bias_b']:+.3f}->{sa['bias_v']:+.3f}"
        r['W-L'] = f"{sa['wins']}-{sa['losses']}"
        yrs = []
        for y in (2022, 2023, 2024, 2025):
            s2 = paired_scores(d, base, pred, window & (d['year'] == y).to_numpy())['START-ALL']
            yrs.append(s2['rmse_v'] - s2['rmse_b'])
        r['yrs better'] = f"{sum(v < 0 for v in yrs)}/4"
        bs = {}
        for yy, pp in params.items():
            for pos, b in pp.items():
                bs.setdefault(pos, []).append(b)
        r['b (held-out range)'] = ' '.join(f"{p} {min(v):+.2f}..{max(v):+.2f}" for p, v in bs.items())
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    pd.set_option('display.width', 230)
    d = load_dump('.sweeps/calibration_lab_v7.csv', '.sweeps/calibration_lab_v7_wk1_ourlads.csv')
    d['shown'] = shipped_board(d)
    feats = luck_features(load_ff())
    d = attach(d, feats)
    played = (d['played'] & d['actual'].notna() & d['week'].between(3, 17)).to_numpy()
    print(f"rows with luck features: {int(d['luck_pg'].notna().sum()):,} of {len(d):,} dump rows "
          f"({d.loc[played, 'luck_pg'].notna().mean():.0%} of played weeks 3-17)")
    print('\n== 1. Does earlier-games luck predict the shipped board miss (actual - shown, points)? weeks 3-17, played ==')
    print(slope_table(d, played).round(3).to_string(index=False))
    d['final'] = d['shown']
    window = ((d['week'].between(3, 17)) & (d['year'] >= 2022)).to_numpy()
    print('== 2. Held-out luck correction (leave-one-season-out b), harness window 2022-2025 weeks 3-17 vs the shipped board ==')
    variants = {}
    for label, positions in (('WR/TE/QB', ('WR', 'TE', 'QB')), ('WR/TE', ('WR', 'TE')), ('all four', ('RB', 'WR', 'TE', 'QB'))):
        for k in (0, 4, 8):
            variants[f'{label}, K={k}'] = loyo_luck(d, positions, k)
    for feat in ('td_luck_pg', 'other_luck_pg'):
        variants[f'WR/TE/QB, {feat}, K=4'] = loyo_luck(d, ('WR', 'TE', 'QB'), 4, feature=feat)
    report(d, variants, window)
    d.to_parquet(os.path.join('.sweeps', 'expected_points_rows.parquet'))


if __name__ == '__main__':
    main()
