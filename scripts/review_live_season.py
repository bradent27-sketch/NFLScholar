"""
In-season review of the live model (2026-10-05): how is it doing on real
weeks, and where does it miss?

PART 1 - LIVE LEDGER (what was really shown before kickoff).
For every team-week, the latest ledger board (data/ledger/) built BEFORE that
team's kickoff is used, so a board saved after Thursday night never scores the
Thursday teams. Boards built after the whole week was played (the retroactive
week 1-2 builds of 2026-09-24) are skipped automatically. Model, market and
FantasyPros are scored on the SAME players: rows where all three have a
projection and the player has a box score.

PART 2 - CURRENT MODEL REBUILT AS-OF EACH WEEK.
Weeks are rebuilt with today's DEFAULT_FEATURES plus the historical injury and
reserve replay, as of that week (no leakage: only games before the week, and
the injury/roster status known that week). Everything after 2025 is out of
sample for every fitted constant. Residuals (projection - actual) are broken
down by position, week, projection tier, stat channel, availability and game
script, scored on players who played (the harness convention), with the
did-not-play rows reported separately.

Usage:
    python scripts/review_live_season.py --year 2026 --weeks 1-4
    python scripts/review_live_season.py --year 2026 --weeks 1-4 --skip-rebuild
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.prediction_ledger import LEDGER_DIR  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from scripts import harness_v2  # noqa: E402

PPR = {'passing_yards': 0.04, 'passing_tds': 4.0, 'passing_interceptions': -2.0, 'rushing_yards': 0.1,
       'rushing_tds': 6.0, 'receptions': 1.0, 'receiving_yards': 0.1, 'receiving_tds': 6.0}
STATS = ['passing_attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_attempts',
         'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']
REPLAY = {'v2_historical_injury_replay', 'v2_historical_reserve_replay'}


def kickoffs_utc(schedule, week):
    """{team: kickoff Timestamp (UTC)} - nflverse gameday/gametime are US Eastern."""
    g = schedule[pd.to_numeric(schedule['week'], errors='coerce') == week]
    out = {}
    for _, r in g.iterrows():
        ts = pd.Timestamp(f"{r['gameday']} {r['gametime']}").tz_localize('America/New_York').tz_convert('UTC')
        for t in (r['home_team'], r['away_team']):
            out[str(t).upper()] = ts
    return out


def actuals(stats_df, name_col, week):
    wk = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week].copy()
    cols = [c for c in STATS if c in wk.columns]
    a = wk.groupby(name_col, observed=True)[cols + ['fantasy_points_ppr']].sum()
    a.index = clean_name_exact(pd.Series(a.index))
    a = a[~a.index.duplicated()]
    team = wk.assign(_k=clean_name_exact(wk[name_col])).drop_duplicates('_k').set_index('_k')['game_team']
    teams_played = set(wk['game_team'].dropna().astype(str).str.upper())
    return a, team, teams_played


# ---------------------------------------------------------------- PART 1
def pregame_ledger(year, week, kick):
    files = sorted(glob.glob(os.path.join(LEDGER_DIR, f'{year}_wk{week:02d}_*.parquet')))
    boards = []
    for p in files:
        d = pd.read_parquet(p)
        boards.append((pd.Timestamp(d['build_ts'].iloc[0]), d))
    rows = []
    for team, ko in kick.items():
        before = [(ts, d) for ts, d in boards if ts < ko]
        if not before:
            continue
        ts, d = max(before, key=lambda x: x[0])
        sub = d[d['Team'].astype(str).str.upper() == team].copy()
        sub['board_ts'] = ts
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def part1(year, weeks, stats_df, name_col, schedule):
    print("\n" + "=" * 100 + "\nPART 1 - LIVE LEDGER, latest board saved before each team's kickoff\n" + "=" * 100)
    frames = []
    for week in weeks:
        kick = kickoffs_utc(schedule, week)
        led = pregame_ledger(year, week, kick)
        if led.empty:
            print(f"  week {week}: no board was saved before any kickoff - skipped")
            continue
        act, _team, played_teams = actuals(stats_df, name_col, week)
        led = led[led['Team'].astype(str).str.upper().isin(played_teams)]
        led['_key'] = clean_name_exact(led['Player'])
        led = led.merge(act.add_prefix('act_'), left_on='_key', right_index=True, how='left')
        led['played'] = led['act_fantasy_points_ppr'].notna()
        led['week'] = week
        fp_col = next((c for c in led.columns if c.startswith('FP Proj Pts')), None)
        led['FP pts'] = led[fp_col] if fp_col else np.nan
        print(f"  week {week}: {led['Team'].nunique()} teams scored from {led['board_ts'].nunique()} pre-kickoff board(s), "
              f"{len(led)} rows, {led['played'].mean():.0%} played | market rows {led['Mkt Market Pts'].notna().sum()} | "
              f"FP rows {led['FP pts'].notna().sum()}")
        frames.append(led)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    pl = d[d['played']]
    common = pl.dropna(subset=['Model Proj Pts', 'Mkt Market Pts', 'FP pts'])
    print(f"\n  Common pool (all three sources + played): {len(common)} player-weeks "
          f"(of {len(pl)} played; market covers {pl['Mkt Market Pts'].notna().mean():.0%})")
    print(f"\n  {'scope':<10}{'n':>5}  " + "  ".join(f"{s:>30}" for s in ('Model', 'Market', 'FantasyPros')))
    print(f"  {'':<15}  " + "  ".join(f"{'RMSE   MAE   bias  pairwise':>30}" for _ in range(3)))
    for pos in (None, 'QB', 'RB', 'WR', 'TE'):
        sub = common if pos is None else common[common['Pos'] == pos]
        for scope, s2 in ((pos or 'ALL', sub),):
            if len(s2) < 10:
                continue
            cells = []
            for col in ('Model Proj Pts', 'Mkt Market Pts', 'FP pts'):
                m = harness_v2.metrics(s2[col].astype(float), s2['act_fantasy_points_ppr'].astype(float))
                cells.append(f"{m['rmse']:6.2f}{m['mae']:6.2f}{m['bias']:+7.2f}{m['pairwise_acc']:9.3f}")
            print(f"  {scope:<10}{len(s2):>5}  " + "  ".join(f"{c:>30}" for c in cells))
    print("\n  Per-stat bias (projection - actual) on the same common pool:")
    for stat in ('rushing_attempts', 'rushing_yards', 'targets', 'receptions', 'receiving_yards', 'passing_yards'):
        cells = []
        for lab, col in (('model', stat), ('market', f'Mkt {stat}'), ('FP', f'FP {stat}')):
            if col in common.columns:
                j = common[[col, f'act_{stat}']].apply(pd.to_numeric, errors='coerce').dropna()
                if len(j) >= 10:
                    cells.append(f"{lab} {(j[col] - j[f'act_{stat}']).mean():+6.2f} (n={len(j)})")
        if cells:
            print(f"    {stat:<18}" + "   ".join(cells))
    # Where model and market disagree most, who was right?
    c = common.copy()
    c['gap'] = c['Model Proj Pts'] - c['Mkt Market Pts']
    c['model_err'] = (c['Model Proj Pts'] - c['act_fantasy_points_ppr']).abs()
    c['mkt_err'] = (c['Mkt Market Pts'] - c['act_fantasy_points_ppr']).abs()
    c['gb'] = pd.cut(c['gap'], [-99, -4, -2, -0.5, 0.5, 2, 4, 99])
    g = c.groupby('gb', observed=True).agg(n=('gap', 'size'), model=('Model Proj Pts', 'mean'), market=('Mkt Market Pts', 'mean'),
                                           actual=('act_fantasy_points_ppr', 'mean'),
                                           model_closer=('model_err', lambda s: np.nan))
    g['model_closer'] = c.groupby('gb', observed=True).apply(lambda x: (x['model_err'] < x['mkt_err']).mean())
    print("\n  When model and market disagree (model - market), who was right?")
    print(g.round(2).to_string())
    return d


# ---------------------------------------------------------------- PART 2
def rebuild(year, weeks, stats_df, name_col, schedule, override_weeks=()):
    """``override_weeks``: weeks for which data/qb1_overrides.csv is honoured. The
    file is season-level (no week column), so today's choices (a backup named
    after an injury) would otherwise be forced onto earlier weeks when the
    starter was healthy. Other weeks run with no override, i.e. the model's own
    QB1 resolution."""
    import data.weekly_projections as wp
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES, game_environment
    feats = frozenset(DEFAULT_FEATURES | REPLAY)
    real_loader = wp.load_qb1_overrides
    frames = []
    for week in weeks:
        act, _team, played_teams = actuals(stats_df, name_col, week)
        wp.load_qb1_overrides = (real_loader if week in override_weeks else
                                 (lambda yr, path=None: (pd.DataFrame(columns=wp.QB1_OVERRIDE_COLUMNS), None)))
        try:
            build_weekly_projections.clear()
            # A week that is not fully played is a LIVE build: the historical replay
            # does not run there, so the live injury feed stands in (as on the real board).
            proj, meta = build_weekly_projections(year, week, 'Full PPR', as_of_week=week,
                                                  apply_injury=(week in override_weeks), features=feats)
            build_weekly_projections.clear()
        finally:
            wp.load_qb1_overrides = real_loader
        proj = proj[proj['Team'].astype(str).str.upper().isin(played_teams)].copy()
        proj = proj[pd.to_numeric(proj['Availability'], errors='coerce').fillna(1.0) > 0.01]
        proj['_key'] = clean_name_exact(proj['Player'])
        proj = proj.merge(act.add_prefix('act_'), left_on='_key', right_index=True, how='left')
        proj['played'] = proj['act_fantasy_points_ppr'].notna()
        env = game_environment(schedule, week) or {}
        g = schedule[pd.to_numeric(schedule['week'], errors='coerce') == week]
        sp = {**dict(zip(g['home_team'], pd.to_numeric(g['spread_line'], errors='coerce'))),
              **dict(zip(g['away_team'], -pd.to_numeric(g['spread_line'], errors='coerce')))}
        proj['spread'] = proj['Team'].map(sp)
        proj['implied'] = proj['Team'].map(lambda t: (env.get(t) or {}).get('implied', np.nan))
        proj['week'] = week
        frames.append(proj)
        print(f"  rebuilt week {week}: {len(proj)} live rows, {proj['played'].mean():.0%} played", flush=True)
    return pd.concat(frames, ignore_index=True)


def part2(d):
    print("\n" + "=" * 100 + "\nPART 2 - CURRENT MODEL REBUILT AS-OF EACH WEEK (replay on), residual = projection - actual\n" + "=" * 100)
    d = d.copy()
    d['proj'] = d['Model Proj Pts'].astype(float)
    d['raw'] = d['Raw Model Proj Pts'].astype(float)
    d['act'] = d['act_fantasy_points_ppr'].astype(float)
    d['pos_rank'] = d.groupby(['week', 'Pos'])['proj'].rank(ascending=False, method='first')
    pl = d[d['played']].copy()
    pl['err'] = pl['proj'] - pl['act']
    start = pl[pl.apply(lambda r: r['pos_rank'] <= harness_v2.STARTABLE_N.get(r['Pos'], 30), axis=1)]

    print("\n  Played players, by position (all / startable).  Reference, harness 2022-2025 wk3-17 with the same"
          " model:\n  START-ALL RMSE 7.70 bias -0.64 | START-QB 7.49/-0.45 RB 7.72/-0.53 WR 7.96/-0.80 TE 6.67/-0.68")
    for pos in (None, 'QB', 'RB', 'WR', 'TE'):
        for lab, sub in (('', pl), ('START-', start)):
            s = sub if pos is None else sub[sub['Pos'] == pos]
            if len(s) < 10:
                continue
            m = harness_v2.metrics(s['proj'], s['act'])
            print(f"    {lab + (pos or 'ALL'):<10} n={m['n']:5d}  RMSE {m['rmse']:5.2f}  MAE {m['mae']:5.2f}  bias {m['bias']:+5.2f}  "
                  f"pairwise {m['pairwise_acc']:.3f}  | raw bias {(s['raw'] - s['act']).mean():+5.2f}")

    print("\n  Startable bias by week (proj - actual):")
    print(start.pivot_table(index='Pos', columns='week', values='err', aggfunc='mean').round(2).to_string())

    print("\n  Bias by projection tier within position (played):")
    pl['tier'] = pd.cut(pl['pos_rank'], [0, 6, 12, 24, 36, 999], labels=['1-6', '7-12', '13-24', '25-36', '37+'])
    t = pl.pivot_table(index='Pos', columns='tier', values='err', aggfunc='mean', observed=True).round(2)
    n = pl.pivot_table(index='Pos', columns='tier', values='err', aggfunc='size', observed=True)
    print(t.to_string()); print("  (n)"); print(n.to_string())

    print("\n  Points bias by stat channel, startable played (PPR points/player-week; - = model low):")
    rows = {}
    for pos in ('QB', 'RB', 'WR', 'TE'):
        s = start[start['Pos'] == pos]
        r = {}
        for stat, w in PPR.items():
            if stat in s.columns and f'act_{stat}' in s.columns:
                r[stat] = w * (s[stat].fillna(0).astype(float) - s[f'act_{stat}'].fillna(0).astype(float)).mean()
        r['calibration (cal - raw)'] = (s['proj'] - s['raw']).mean()
        r['TOTAL'] = s['err'].mean()
        rows[pos] = r
    print(pd.DataFrame(rows).round(2).to_string())

    print("\n  Volume and efficiency, startable played (proj / actual):")
    for pos, pairs in (('QB', [('passing_attempts', None), ('passing_yards', 'passing_attempts'), ('passing_tds', 'passing_attempts'),
                               ('rushing_attempts', None), ('rushing_yards', 'rushing_attempts')]),
                       ('RB', [('rushing_attempts', None), ('rushing_yards', 'rushing_attempts'), ('rushing_tds', 'rushing_attempts'),
                               ('targets', None), ('receptions', 'targets'), ('receiving_yards', 'targets')]),
                       ('WR', [('targets', None), ('receptions', 'targets'), ('receiving_yards', 'targets'), ('receiving_tds', 'targets')]),
                       ('TE', [('targets', None), ('receptions', 'targets'), ('receiving_yards', 'targets'), ('receiving_tds', 'targets')])):
        s = start[start['Pos'] == pos]
        cells = []
        for stat, per in pairs:
            p, a = s[stat].astype(float).sum(), s[f'act_{stat}'].astype(float).sum()
            if per is None:
                cells.append(f"{stat} {p / len(s):.2f}/{a / len(s):.2f}")
            else:
                pp, aa = s[per].astype(float).sum(), s[f'act_{per}'].astype(float).sum()
                cells.append(f"{stat}/{per.split('_')[0][:4]} {p / pp:.3f}/{a / aa:.3f}")
        print(f"    {pos}: " + " | ".join(cells))

    print("\n  Startable bias by game context (team spread, + = favored):")
    start = start.copy()
    start['sb'] = pd.cut(start['spread'], [-30, -6.5, -2.5, 2.5, 6.5, 30], labels=['dog 7+', 'dog 3-6', 'pick', 'fav 3-6', 'fav 7+'])
    print(start.pivot_table(index='Pos', columns='sb', values='err', aggfunc='mean', observed=True).round(2).to_string())

    print("\n  Questionable / limited (0.01 < availability < 1) vs healthy, played rows:")
    pl['avail_b'] = np.where(pd.to_numeric(pl['Availability'], errors='coerce') < 0.99, 'limited', 'healthy')
    print(pl.groupby(['Pos', 'avail_b'])['err'].agg(['size', 'mean']).round(2).unstack().to_string())

    dnp = d[~d['played']]
    print(f"\n  Did not play (no box score, listed live): {len(dnp)} rows, projected {dnp['proj'].sum():.0f} pts total; "
          f"startable DNP: {int((dnp['pos_rank'] <= dnp['Pos'].map(harness_v2.STARTABLE_N).fillna(30)).sum())}")
    top = dnp.sort_values('proj', ascending=False).head(8)
    print("   largest:", "; ".join(f"{r.Player} ({r.Pos} wk{r.week}, {r.proj:.1f})" for r in top.itertuples()))

    print("\n  Team-level: model total for a team's skill players vs actual (startable + played), largest misses:")
    tm = pl.groupby(['week', 'Team']).agg(proj=('proj', 'sum'), act=('act', 'sum'), implied=('implied', 'first'))
    tm['err'] = tm['proj'] - tm['act']
    print(f"    team-week skill-points bias {tm['err'].mean():+.2f}, sd {tm['err'].std():.1f}; corr(err, implied) "
          f"{tm[['err', 'implied']].corr().iloc[0, 1]:+.2f}")
    print(tm.reindex(tm['err'].abs().sort_values(ascending=False).index).head(8).round(1).to_string())

    print("\n  Biggest individual misses (startable, played):")
    s = start.reindex(start['err'].abs().sort_values(ascending=False).index).head(12)
    print(s[['week', 'Player', 'Pos', 'Team', 'proj', 'act', 'err']].round(1).to_string(index=False))
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--weeks', default='1-4')
    ap.add_argument('--skip-rebuild', action='store_true')
    ap.add_argument('--out', default='.sweeps/live_review_2026')
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.weeks.split('-'))
    weeks = list(range(lo, hi + 1))
    pd.set_option('display.width', 220)
    from data.loaders import load_schedule
    stats_df, _tc, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    schedule = load_schedule(a.year)
    os.makedirs(a.out, exist_ok=True)
    led = part1(a.year, weeks, stats_df, name_col, schedule)
    if led is not None:
        led.to_parquet(os.path.join(a.out, 'ledger_pregame.parquet'), index=False)
    if not a.skip_rebuild:
        print("\nrebuilding with the current model ...")
        rb = rebuild(a.year, weeks, stats_df, name_col, schedule, override_weeks={max(weeks)})
        rb.to_parquet(os.path.join(a.out, 'rebuild.parquet'), index=False)
    else:
        rb = pd.read_parquet(os.path.join(a.out, 'rebuild.parquet'))
    part2(rb)


if __name__ == '__main__':
    main()
