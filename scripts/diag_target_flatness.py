"""
Diagnostic for the clean-room WR/TE target-share flatness (known issue in
docs/weekly_projections_methodology.md, 2026-09-25): in a WR/TE room with
nobody OUT, the room's target TOTAL is about right but ranks 1-3 are
under-projected and ranks 6-8 over-projected.

The pass-capacity fit is ONE uniform factor per room, so it cannot change any
player's SHARE of his room. This script locates the stage that does, from the
per-player trace build_weekly_projections already records in
meta['explanations'][...]['stats']['targets']:

  blended_rate            own in-season rate blended with a prior (player's own
                          prior season, or a position-average-per-snap fallback)
  pre_vacancy_projection  after matchup / script / pace / environment
  targets (board)         after vacancy + pass-capacity conservation

One arm only (DEFAULT_FEATURES + historical injury replay, so OUT players are
zeroed and a room with a replayed absence can be excluded as not clean).

Usage:
    python scripts/diag_target_flatness.py --years 2022,2023,2024,2025 --weeks 3-17 --out .sweeps/diag_flat
    python scripts/diag_target_flatness.py --summarize --out .sweeps/diag_flat
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPLAY = 'v2_historical_injury_replay'
TRACE_FIELDS = ('current_rate', 'raw_prior_rate', 'prior_rate', 'prior_source', 'current_weight',
                'current_games', 'role_scale', 'expected_snap_share', 'prior_snap_share',
                'blended_rate', 'matchup_multiplier', 'script_multiplier', 'script_neutral_multiplier',
                'pace_multiplier', 'environment_multiplier', 'availability_multiplier',
                'pre_vacancy_projection', 'role_confidence', 'prior2_weight')


def collect(years, weeks, out_dir, extra_features=()):
    from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES
    from data.transforms import load_and_merge_data

    feats = frozenset(DEFAULT_FEATURES | {REPLAY} | set(extra_features))
    frames = []
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
        s = stats_df.copy()
        s['week'] = pd.to_numeric(s['week'], errors='coerce')
        s['targets'] = pd.to_numeric(s['targets'], errors='coerce').fillna(0.0)
        s['weekly_snap_pct'] = pd.to_numeric(s.get('weekly_snap_pct'), errors='coerce')
        # Name-keyed, same convention as the other diag scripts.
        act = (s.groupby([name_col, 'week'])
               .agg(act_targets=('targets', 'sum'), act_snap_pct=('weekly_snap_pct', 'max'))
               .reset_index().rename(columns={name_col: 'Player'}))
        act['played'] = True
        for week in weeks:
            if not (s['week'] == week).any():
                continue
            proj, meta = build_weekly_projections(
                year, week, 'Full PPR', as_of_week=week, apply_injury=False, features=feats)
            if proj.empty:
                continue
            out_rooms = set()
            for entry in meta.get('vacancy_ledger') or []:
                if entry.get('volume') == 'targets' and str(entry.get('functional_source_role', '')) in ('WR', 'TE'):
                    out_rooms.add(str(entry.get('team', '')).upper())
            expl = meta.get('explanations') or {}
            recs = []
            for _, r in proj[proj['Pos'].isin(['WR', 'TE'])].iterrows():
                t = ((expl.get((r['Player'], r['Pos'], r['Team'])) or {}).get('stats') or {}).get('targets') or {}
                rec = {'year': year, 'week': week, 'Player': r['Player'], 'Pos': r['Pos'],
                       'Team': str(r['Team']).upper(), 'targets': float(r.get('targets', np.nan)),
                       'Availability': float(r.get('Availability', np.nan)),
                       'Model Proj Pts': float(r.get('Model Proj Pts', np.nan))}
                for f in TRACE_FIELDS:
                    rec[f] = t.get(f, np.nan)
                recs.append(rec)
            frame = pd.DataFrame(recs)
            if frame.empty:
                continue
            frame['room_has_out'] = frame['Team'].isin(out_rooms)
            frame = frame.merge(act[act['week'] == week].drop(columns='week'), on='Player', how='left')
            frame['played'] = frame['played'].fillna(False).astype(bool)
            frame['act_targets'] = frame['act_targets'].fillna(0.0)
            frames.append(frame)
            print(f"{year} w{week} done ({len(frame)} WR/TE rows)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, 'wrte.parquet'), index=False)
    return df


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)) if len(x) > 2 else np.nan


def summarize(df, strict=True):
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)
    keys = ['year', 'week', 'Team']
    d = df[~df['room_has_out']].copy()
    # Sidelined rows (replay OUT) project ~0 and aren't part of the question.
    d = d[d['Availability'] > 0.01]
    for c in ('blended_rate', 'pre_vacancy_projection', 'targets'):
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    d = d[d.groupby(keys)['act_targets'].transform('sum') > 0]
    if strict:
        # The injury replay misses IR/PUP/suspension absences (e.g. 2024 wk8
        # Higgins, Collins, Hockenson, Jameson Williams all Availability 1.0),
        # so a replay-"clean" room can still have lost a real contributor and
        # redistributed his volume. Hindsight filter, diagnosis only: drop any
        # room where a player projected >= 1.5 targets didn't play.
        missed = (~d['played']) & (d['targets'] >= 1.5)
        d = d[~missed.groupby([d[k] for k in keys]).transform('any')]
    print(f"{'strict ' if strict else ''}clean WR/TE rooms: {d.groupby(keys).ngroups} team-weeks, "
          f"{len(d)} player-rows")

    d['rank'] = d.groupby(keys)['targets'].rank(ascending=False, method='first')
    d['rb'] = pd.cut(d['rank'], [0, 1, 2, 3, 5, 8, 99], labels=['1', '2', '3', '4-5', '6-8', '9+'])
    d['act_share'] = d['act_targets'] / d.groupby(keys)['act_targets'].transform('sum')
    stages = {'blend': 'blended_rate', 'pre_cap': 'pre_vacancy_projection', 'final': 'targets'}
    for name, col in stages.items():
        d[f'sh_{name}'] = d[col] / d.groupby(keys)[col].transform('sum')

    print("\n=== share slope (actual share ~ projected share; 1.0 = right spread, >1 = too flat) ===")
    for name in stages:
        print(f"  {name:8s} {_slope(d[f'sh_{name}'], d['act_share']):.3f}")

    print("\n=== by projected room rank: mean targets and shares per stage ===")
    agg = {'n': ('targets', 'size'), 'act': ('act_targets', 'mean'), 'final': ('targets', 'mean'),
           'pre_cap': ('pre_vacancy_projection', 'mean'), 'played': ('played', 'mean'),
           'act_sh': ('act_share', 'mean')}
    agg.update({f'sh_{n}': (f'sh_{n}', 'mean') for n in stages})
    g = d.groupby('rb', observed=True).agg(**agg)
    g['final_err'] = g['final'] - g['act']
    print(g.round(3).to_string())

    print("\n=== by prior source (player prior vs position fallback) ===")
    d['psrc'] = d['prior_source'].fillna('?')
    for src, s in d.groupby('psrc'):
        print(f"  {src:20s} n={len(s):6d}  share slope(final) {_slope(s['sh_final'], s['act_share']):.3f}  "
              f"target bias {(s['targets'] - s['act_targets']).mean():+.3f}")
    g = d.groupby(['rb', 'psrc'], observed=True).agg(n=('targets', 'size'), proj=('targets', 'mean'),
                                                      act=('act_targets', 'mean'))
    g['err'] = g['proj'] - g['act']
    print(g.round(3).to_string())

    print("\n=== played vs not (tail over-projection from inactives?) ===")
    g = d.groupby(['rb', 'played'], observed=True).agg(n=('targets', 'size'), proj=('targets', 'mean'),
                                                       act=('act_targets', 'mean'))
    g['err'] = g['proj'] - g['act']
    print(g.round(3).to_string())

    print("\n=== blend components among players with BOTH an own in-season rate and a player prior ===")
    b = d[(d['psrc'] == 'player prior') & np.isfinite(pd.to_numeric(d['current_rate'], errors='coerce'))].copy()
    for c in ('current_rate', 'prior_rate', 'current_weight', 'current_games'):
        b[c] = pd.to_numeric(b[c], errors='coerce')
    b['gb'] = pd.cut(b['current_games'], [-0.1, 2, 4, 6, 9, 20], labels=['0-2', '3-4', '5-6', '7-9', '10+'])
    # Optimal weight on the own in-season rate per games bucket, by least squares on
    # (current - prior) -> (actual - prior), then compare to the model's own weight.
    rows = []
    for gb, s in b.groupby('gb', observed=True):
        x = (s['current_rate'] - s['prior_rate']).to_numpy()
        y = (s['act_targets'] - s['prior_rate']).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        w_opt = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()) if ok.sum() > 10 else np.nan
        rows.append({'games': gb, 'n': int(ok.sum()), 'model_w': s['current_weight'].mean(), 'lsq_w': w_opt,
                     'slope_blend': _slope(s['blended_rate'], s['act_targets']),
                     'slope_cur': _slope(s['current_rate'], s['act_targets']),
                     'slope_prior': _slope(s['prior_rate'], s['act_targets'])})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== snap-share channel: projected expected_snap_share vs actual snap % by rank ===")
    if 'act_snap_pct' in d.columns:
        print(d.groupby('rb', observed=True).agg(exp_snap=('expected_snap_share', 'mean'),
                                                   act_snap=('act_snap_pct', 'mean')).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/diag_flat')
    ap.add_argument('--features', default='', help='comma-separated extra flags for the arm')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        df = pd.read_parquet(os.path.join(args.out, 'wrte.parquet'))
    else:
        years = [int(y) for y in args.years.split(',')]
        lo, hi = (int(x) for x in args.weeks.split('-'))
        extra = [f for f in args.features.split(',') if f]
        df = collect(years, list(range(lo, hi + 1)), args.out, extra)
    print("########## replay-clean rooms ##########")
    summarize(df, strict=False)
    print("\n########## strict-clean rooms (every >=1.5-target player played) ##########")
    summarize(df, strict=True)


if __name__ == '__main__':
    main()
