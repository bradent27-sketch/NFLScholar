"""
RB level audit (2026-09-30). The calibration re-fit found startable RBs who
played under-projected on every channel (2021-2025 wk5+: rush yards -5.0, TDs
-0.07, targets -0.40, receptions -0.29, receiving yards -1.87, points -1.43).
Before any calibration line moves, this separates:

  * SURVIVORSHIP: a depth back discounted by P(play) has a box score only when
    he played, so a played-only pool is biased low by construction. Every row
    (DNP = 0) is kept here, with the played-only view beside it.
  * WHERE in a room the gap sits (RB1 / RB2 / RB3 / RB4+ by projected carries),
    per channel, so a level error is not confused with a depth-back artifact.
  * WHICH channel and stage: carries vs yards/carry vs TDs vs targets vs
    catch rate vs receiving yards, and the per-stage trace (blended rate ->
    multipliers -> vacancy/budget/capacity) so the stage that loses the level
    can be named.
  * WHAT else is in the actual points (fumbles, 2-pt conversions) that the
    projected stat line never contains.

Collect (per-week parquet files are written as it goes, so a partial run can be
analyzed):
    python scripts/diag_rb_level_audit.py collect --years 2022,2023,2024,2025 --weeks 4,6,8,10,12,14,16 --out .sweeps/rb_level_audit
Analyze:
    python scripts/diag_rb_level_audit.py analyze --out .sweeps/rb_level_audit
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPLAY = 'v2_historical_injury_replay'
RESERVE = 'v2_historical_reserve_replay'
STATS = ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds')
ACT_COL = {'rushing_attempts': 'rushing_attempts', 'rushing_yards': 'rushing_yards', 'rushing_tds': 'rushing_tds',
           'targets': 'targets', 'receptions': 'receptions', 'receiving_yards': 'receiving_yards',
           'receiving_tds': 'receiving_tds'}
TRACE_FIELDS = ('blended_rate', 'matchup_multiplier', 'script_multiplier', 'script_neutral_multiplier',
                'pace_multiplier', 'environment_multiplier', 'availability_multiplier', 'participation_multiplier',
                'pre_vacancy_projection', 'carry_budget_delta', 'pass_capacity_delta', 'vacancy_delta',
                'final_projection')
LABEL = {'matchup_multiplier': 'matchup', 'script_multiplier': 'script', 'script_neutral_multiplier': 'descript',
         'pace_multiplier': 'pace', 'environment_multiplier': 'env', 'participation_multiplier': 'partic'}
PPR = {'rushing_yards': 0.1, 'rushing_tds': 6.0, 'receptions': 1.0, 'receiving_yards': 0.1, 'receiving_tds': 6.0}


def collect(years, weeks, out_dir, features_extra=(), drop=()):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data
    feats = frozenset((DEFAULT_FEATURES | {REPLAY, RESERVE} | set(features_extra)) - set(drop))
    os.makedirs(out_dir, exist_ok=True)
    for year in years:
        stats_df, team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df.copy()
        s['_w'] = pd.to_numeric(s['week'], errors='coerce')
        for c in list(ACT_COL.values()) + ['fantasy_points_ppr', 'weekly_snap_pct']:
            s[c] = pd.to_numeric(s.get(c), errors='coerce')
        rb_stats = s[s['position'].astype(str) == 'RB']
        for week in weeks:
            wk = rb_stats[rb_stats['_w'] == week]
            if wk.empty:
                continue
            act = wk.groupby(name_col).agg(**{f'act_{k}': (v, 'sum') for k, v in ACT_COL.items()},
                                           act_pts=('fantasy_points_ppr', 'sum'), act_snap=('weekly_snap_pct', 'max'))
            act['played'] = True
            proj, meta = build_weekly_projections(year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
            if proj.empty:
                continue
            expl = meta.get('explanations') or {}
            rows = []
            for _, r in proj[proj['Pos'] == 'RB'].iterrows():
                st = ((expl.get((r['Player'], 'RB', r['Team'])) or {}).get('stats') or {})
                rec = {'year': year, 'week': week, 'Player': r['Player'], 'Team': str(r['Team']).upper(),
                       'Availability': float(r.get('Availability', np.nan)),
                       'raw_pts': float(r.get('Raw Model Proj Pts', np.nan)), 'pts': float(r.get('Model Proj Pts', np.nan)),
                       'snap_share': float(r.get('Expected Snap Share', np.nan))}
                for stat in STATS:
                    rec[f'p_{stat}'] = float(r.get(stat, np.nan))
                    t = st.get(stat) or {}
                    for f in TRACE_FIELDS:
                        rec[f'{stat}.{f}'] = t.get(f, np.nan)
                rows.append(rec)
            fr = pd.DataFrame(rows).merge(act, left_on='Player', right_index=True, how='left')
            fr['played'] = fr['played'].fillna(False).astype(bool)
            for c in [c for c in fr.columns if c.startswith('act_') and c != 'act_snap']:
                fr[c] = fr[c].fillna(0.0)
            vac = {str(e.get('team', '')).upper() for e in (meta.get('vacancy_ledger') or []) if e.get('volume') == 'rushing_attempts'}
            fr['room_has_out'] = fr['Team'].isin(vac)
            fr.to_parquet(os.path.join(out_dir, f'rb_{year}_w{week:02d}.parquet'), index=False)
            print(f"{year} w{week} done ({len(fr)} RBs)", flush=True)


def load(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, 'rb_*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    keys = ['year', 'week', 'Team']
    live = d['Availability'] > 0.01
    d['live'] = live
    d['rank'] = d[live].groupby(keys)['p_rushing_attempts'].rank(ascending=False, method='first').reindex(d.index)
    d['rk'] = pd.cut(d['rank'], [0, 1, 2, 3, 99], labels=['RB1', 'RB2', 'RB3', 'RB4+'])
    # Projected / actual points rebuilt from the stat lines (PPR), plus the
    # part of actual points no projected channel covers (fumbles, 2-pt).
    d['pts_from_proj'] = sum(d[f'p_{k}'] * w for k, w in PPR.items())
    d['pts_from_act_stats'] = sum(d[f'act_{k}'] * w for k, w in PPR.items())
    d['act_other'] = d['act_pts'] - d['pts_from_act_stats']
    return d


def _bias_table(d, label, cols):
    rows = []
    for c in cols:
        p, a = d[f'p_{c}'], d[f'act_{c}']
        rows.append({'stat': c, 'proj': p.mean(), 'act': a.mean(), 'bias': (p - a).mean()})
    return pd.DataFrame(rows).set_index('stat').round(3)


def analyze(out_dir):
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)
    d = load(out_dir)
    d = d[d['live']].copy()
    print(f"rows {len(d)} live RBs  weeks {sorted(d['week'].unique())} years {sorted(d['year'].unique())}  "
          f"played {d['played'].mean():.1%}")
    chan = list(STATS)

    print("\n=== 1. SURVIVORSHIP: startable pool (top-N RBs per week by projected points), every row vs played-only ===")
    d['pool_rank'] = d.groupby(['year', 'week'])['raw_pts'].rank(ascending=False, method='first')
    for n in (24, 36, 48):
        pool = d[d['pool_rank'] <= n]
        for lab, sub in (('ALL rows (DNP = 0)', pool), ('PLAYED only', pool[pool['played']])):
            e = sub['raw_pts'] - sub['act_pts']
            print(f"  top-{n:<3d} {lab:22s} n={len(sub):5d}  raw pts bias {e.mean():+.2f}  "
                  f"(proj {sub['raw_pts'].mean():.2f} act {sub['act_pts'].mean():.2f})   played {sub['played'].mean():.1%}")

    print("\n=== 2. BIAS BY ROOM RANK, every row vs played-only (raw points and projected-stat channels) ===")
    for lab, sub in (('ALL rows (DNP = 0)', d), ('PLAYED only', d[d['played']])):
        print(f"\n--- {lab}")
        g = sub.groupby('rk', observed=True)
        out = pd.DataFrame({'n': g.size(), 'played%': g['played'].mean() * 100,
                            'raw pts bias': g.apply(lambda x: (x['raw_pts'] - x['act_pts']).mean()),
                            'cal pts bias': g.apply(lambda x: (x['pts'] - x['act_pts']).mean())})
        for c in chan:
            out[f'{c[:9]}'] = g.apply(lambda x, c=c: (x[f'p_{c}'] - x[f'act_{c}']).mean())
        print(out.round(2).to_string())

    print("\n=== 3. CLEAN view: RB1/RB2 who PLAYED and were not in a room with an OUT back (no survivorship, no vacancy) ===")
    core = d[(d['rank'] <= 2) & d['played'] & ~d['room_has_out']]
    print(f"n={len(core)}")
    print(_bias_table(core, 'core', chan).to_string())
    print("  raw points: proj %.2f act %.2f bias %+.2f | act_other (fumbles/2pt not projected) mean %+.3f" % (
        core['raw_pts'].mean(), core['act_pts'].mean(), (core['raw_pts'] - core['act_pts']).mean(), core['act_other'].mean()))
    print("  points rebuilt from projected stat line: %.2f (Raw Model Proj Pts %.2f)" % (core['pts_from_proj'].mean(), core['raw_pts'].mean()))

    print("\n=== 4. EFFICIENCY decomposition, played RBs, by room rank (yards = carries x YPC) ===")
    pl = d[d['played'] & ~d['room_has_out']]
    rows = []
    for lab, sub in list(pl.groupby('rk', observed=True)) + [('ALL', pl)]:
        pc, ac = sub['p_rushing_attempts'].sum(), sub['act_rushing_attempts'].sum()
        py, ay = sub['p_rushing_yards'].sum(), sub['act_rushing_yards'].sum()
        ptd, atd = sub['p_rushing_tds'].sum(), sub['act_rushing_tds'].sum()
        rows.append({'rank': lab, 'n': len(sub), 'carries p/a': f"{pc / len(sub):.2f}/{ac / len(sub):.2f}",
                     'YPC proj': py / pc, 'YPC act': ay / ac,
                     'yds gap/player': (py - ay) / len(sub),
                     '  from carries': (pc - ac) * (ay / ac) / len(sub),
                     '  from YPC': pc * (py / pc - ay / ac) / len(sub),
                     'TD/carry proj': ptd / pc, 'TD/carry act': atd / ac})
    print(pd.DataFrame(rows).set_index('rank').round(3).to_string())

    print("\n=== 5. RECEIVING decomposition, played RBs, by room rank ===")
    rows = []
    for lab, sub in list(pl.groupby('rk', observed=True)) + [('ALL', pl)]:
        pt, at = sub['p_targets'].sum(), sub['act_targets'].sum()
        pr, ar = sub['p_receptions'].sum(), sub['act_receptions'].sum()
        py, ay = sub['p_receiving_yards'].sum(), sub['act_receiving_yards'].sum()
        rows.append({'rank': lab, 'targets p/a': f"{pt / len(sub):.2f}/{at / len(sub):.2f}",
                     'catch% proj': pr / pt, 'catch% act': ar / at, 'Y/rec proj': py / pr, 'Y/rec act': ay / ar,
                     'rec yds gap/player': (py - ay) / len(sub)})
    print(pd.DataFrame(rows).set_index('rank').round(3).to_string())

    print("\n=== 6. CONDITIONAL LEVEL: played RBs, carries/targets projected vs actual by projection bin ===")
    for stat in ('rushing_attempts', 'targets', 'rushing_yards', 'receiving_yards'):
        x = pl.copy()
        x['bin'] = pd.qcut(x[f'p_{stat}'].rank(method='first'), 6, duplicates='drop')
        g = x.groupby('bin', observed=True).agg(proj=(f'p_{stat}', 'mean'), act=(f'act_{stat}', 'mean'))
        g['ratio'] = g['act'] / g['proj']
        print(f"  {stat}: " + "  ".join(f"{p:.1f}->{a:.1f} (x{r:.2f})" for p, a, r in zip(g['proj'], g['act'], g['ratio'])))

    print("\n=== 7. STAGE TRACE, played RB1/RB2 (carry-weighted multipliers; blended rate -> pre-vacancy -> final) ===")
    for stat in ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receiving_yards'):
        w = core['p_rushing_attempts'] if stat.startswith('rush') else core['p_targets']
        w = w.clip(lower=1e-6)
        mult = {f: np.average(core[f'{stat}.{f}'].fillna(1.0), weights=w) for f in
                ('matchup_multiplier', 'script_multiplier', 'script_neutral_multiplier', 'pace_multiplier',
                 'environment_multiplier', 'participation_multiplier')}
        print(f"  {stat:17s} blended {core[f'{stat}.blended_rate'].mean():7.3f}  pre-vac {core[f'{stat}.pre_vacancy_projection'].mean():7.3f}  "
              f"final {core[f'p_{stat}'].mean():7.3f}  actual {core[f'act_{stat}'].mean():7.3f}  | " +
              " ".join(f"{LABEL[k]} {v:.3f}" for k, v in mult.items()) +
              f" | budget d {core[f'{stat}.carry_budget_delta'].fillna(0).mean():+.3f} capacity d {core[f'{stat}.pass_capacity_delta'].fillna(0).mean():+.3f}"
              f" vacancy d {core[f'{stat}.vacancy_delta'].fillna(0).mean():+.3f}")

    print("\n=== 8. By season, played RB1/RB2: raw points bias and carries/yards/targets bias ===")
    g = core.groupby('year')
    print(pd.DataFrame({'n': g.size(), 'raw pts bias': g.apply(lambda x: (x['raw_pts'] - x['act_pts']).mean()),
                        'carries': g.apply(lambda x: (x['p_rushing_attempts'] - x['act_rushing_attempts']).mean()),
                        'rush yds': g.apply(lambda x: (x['p_rushing_yards'] - x['act_rushing_yards']).mean()),
                        'targets': g.apply(lambda x: (x['p_targets'] - x['act_targets']).mean()),
                        'rec yds': g.apply(lambda x: (x['p_receiving_yards'] - x['act_receiving_yards']).mean())}).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['collect', 'analyze'])
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='4,6,8,10,12,14,16')
    ap.add_argument('--out', default='.sweeps/rb_level_audit')
    ap.add_argument('--features', default='')
    ap.add_argument('--drop', default='')
    a = ap.parse_args()
    if a.mode == 'collect':
        collect([int(v) for v in a.years.split(',')], [int(v) for v in a.weeks.split(',')], a.out,
                [f for f in a.features.split(',') if f], [f for f in a.drop.split(',') if f])
    else:
        analyze(a.out)


if __name__ == '__main__':
    main()
