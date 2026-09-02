"""Fixed-weight alignment/scheme blend sweep - REWRITE of
scripts/sweep_scheme_blend_weight.py, which never produced a usable result.

WHY THE OLD ONE NEVER FINISHED (diagnosed 2026-09-01, it was not a hang):

  1. It delegated to backtest_stat_level.evaluate_stat_level ONCE PER WEIGHT,
     and that function rebuilds the BASE projection for every week inside
     every call. Five weights x 3 years x 14 weeks x 2 builds = 420 model
     builds, ~7h, and print_report only fires after a whole weight finishes -
     so there is no output at all for the first ~85 minutes. It was killed at
     2h15m looking dead when it was merely silent and doing 2x the work.

  2. Far worse: it was pointed at --years 2023,2024,2025, and the scheme
     feature is INERT for 2023 and 2024. The player-side man/zone route share
     comes from load_weekly_scheme_profiles, which reads the PFF WEEKLY
     archive (pff_imports/{year}/weekly/) - and at the time only 2025 had
     one. On 2024 wk6 every one of 747 WR/TE scheme lookups returned "Player
     man-coverage route share is unavailable", both scheme flags produce
     byte-identical output to base, and two thirds of the run was grinding
     full model builds that could not differ from baseline by construction.
     The season totals in pff_imports/{year}/*.csv are NOT a substitute:
     they are season aggregates, so using them mid-season leaks the future,
     and unlike the alignment side there is no season-prior rollover for the
     player scheme profile (weekly_projections.py says so at its
     pff_scheme_profiles construction).

2024's weekly archive was backfilled 2026-09-01 and verified loading
time-valid (wk8 2024: 172 of 487 rows move, WR 114/200, TE 58/106), so a
real CROSS-SEASON test is now possible. 2023 remains un-backfilled.

WHAT THIS ONE DOES DIFFERENTLY:

  - Runs only seasons whose weekly archive exists, and REFUSES to quietly
    average in a season where the feature cannot act (see the INERT check).
  - Base built ONCE per week and reused across every weight: 1 + N builds per
    week instead of 2N.
  - Per-week progress, flushed, so a long run is visibly alive.
  - TE and WR swept SEPARATELY. They are NOT independent: measured on 2025
    wk8, changing only the WR weight moved 34 of 90 TE rows by up to 0.4 pts,
    because the pass-capacity allocator redistributes team target capacity
    across positions. Sweeping both in one build to save time would have
    quietly cross-contaminated every number.
  - With 2+ seasons: a PER-SEASON breakdown, not a pooled number. A weight
    that is real holds its sign in each season independently - which is
    exactly what the coaching cohort tables failed to do three times.
    Pooling can let one season carry an effect the other contradicts.
  - With 1 season: falls back to an odd/even WEEK split, clearly labelled
    as the weak substitute it is.

    python scripts/sweep_scheme_blend_weight2.py --position TE --weights 0.6,0.7,0.8,0.9,1.0
    python scripts/sweep_scheme_blend_weight2.py --position WR --weights 0.3,0.4,0.5,0.7
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N  # noqa: E402
from scripts.backtest_component import _bootstrap_ci, _sign_test_p  # noqa: E402
from scripts.backtest_stat_level import _actual_stat  # noqa: E402

BLEND_FLAG = 'v2_scheme_alignment_blend'


def _pos_frames(df, pos):
    d = df[df['Pos'] == pos]
    return d, d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--position', required=True, choices=['WR', 'TE'])
    ap.add_argument('--weights', required=True,
                    help='comma-separated scheme-side weights in [0,1]')
    ap.add_argument('--stats', default='targets,receptions,receiving_yards')
    # 2025 only by default and on purpose - see the module docstring. Left as
    # a flag so a backfilled archive can be swept without editing the script.
    ap.add_argument('--years', default='2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    lo, hi = (args.weeks.split('-') + [None])[:2]
    weeks = list(range(int(lo), int(hi) + 1)) if hi else [int(w) for w in args.weeks.split(',')]
    stats = [s.strip() for s in args.stats.split(',') if s.strip()]
    weights = [float(w) for w in args.weights.split(',') if w.strip()]
    pos = args.position

    print(f"position={pos} weights={weights}")
    print(f"years={years} weeks={weeks[0]}-{weeks[-1]} stats={stats}")
    _ARCHIVED = (2024, 2025)   # seasons with a PFF weekly archive on disk
    missing = [y for y in years if y not in _ARCHIVED]
    if missing:
        print(f"!! WARNING: no PFF weekly scheme archive for {missing} - "
              "those seasons contribute ZERO signal (feature inert).")
    print(f"builds ~= {len(years) * len(weeks) * (1 + len(weights))}\n", flush=True)

    variant = frozenset(DEFAULT_FEATURES | {BLEND_FLAG})
    # rows[w][stat][scope] = list of (metrics_base, metrics_var, week)
    rows = {w: {s: {} for s in stats} for w in weights}
    active_weeks = 0

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, args.scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data, skipped", flush=True)
            continue
        for week in weeks:
            actuals = {s: _actual_stat(stats_df, name_col, week, s) for s in stats}
            if all(a.empty for a in actuals.values()):
                continue

            # Base ONCE per week, reused by every weight below.
            wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
            build_weekly_projections.clear()
            base, bmeta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week,
                apply_injury=False, features=DEFAULT_FEATURES)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})", flush=True)
                continue

            moved_any = False
            for w in weights:
                wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
                wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT[pos] = w
                build_weekly_projections.clear()
                var, _vm = build_weekly_projections(
                    year, week, args.scoring, as_of_week=week,
                    apply_injury=False, features=variant)
                if var.empty:
                    continue
                pool = sorted(set(base['Player']) & set(var['Player']))
                if len(pool) < 20:
                    continue
                b_all = base[base['Player'].isin(pool)]
                v_all = var[var['Player'].isin(pool)]

                # Did this weight actually change anything? A silently inert
                # run is the exact failure that wasted the last attempt, so
                # it is detected and reported instead of averaging to zero.
                bp = b_all.set_index('Player')['Model Proj Pts']
                vp = v_all.set_index('Player')['Model Proj Pts'].reindex(bp.index)
                if (bp - vp).abs().max() > 1e-9:
                    moved_any = True

                b_pos, b_top = _pos_frames(b_all, pos)
                v_pos, v_top = _pos_frames(v_all, pos)
                for stat in stats:
                    actual = actuals[stat]
                    if actual.empty or stat not in b_pos.columns:
                        continue
                    for scope, bdf, vdf in ((pos, b_pos, v_pos),
                                            (f'START-{pos}', b_top, v_top)):
                        pb = pd.Series(bdf[stat].to_numpy(dtype=float), index=bdf['Player'])
                        pv = pd.Series(vdf[stat].to_numpy(dtype=float), index=vdf['Player'])
                        mb, mv = _metrics(pb, actual), _metrics(pv, actual)
                        if mb and mv:
                            rows[w][stat].setdefault(scope, []).append((mb, mv, week, year))

            wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
            build_weekly_projections.clear()
            active_weeks += int(moved_any)
            print(f"{year} w{week} done{'' if moved_any else '  [INERT - no row moved]'}",
                  flush=True)

    _report(pos, weights, stats, rows, active_weeks)


def _block(pairs):
    if not pairs:
        return None
    mb = [p[0] for p in pairs]
    mv = [p[1] for p in pairs]
    wts = [m['n'] for m in mb]
    d = [b['mae'] - a['mae'] for a, b in [(p[0], p[1]) for p in pairs]]
    lo, hi = _bootstrap_ci(d, wts)
    wins = sum(1 for p in pairs if p[1]['mae'] < p[0]['mae'])
    losses = sum(1 for p in pairs if p[0]['mae'] < p[1]['mae'])
    return {
        'n': sum(wts), 'mae_b': _weighted(mb, 'mae'), 'mae_v': _weighted(mv, 'mae'),
        'd': _weighted(mv, 'mae') - _weighted(mb, 'mae'),
        'lo': lo, 'hi': hi, 'sig': np.isfinite(lo) and (lo > 0 or hi < 0),
        'wins': wins, 'losses': losses, 'p': _sign_test_p(wins, losses),
    }


def _report(pos, weights, stats, rows, active_weeks):
    bar = '=' * 100
    print(f"\n\n{bar}\nFIXED SCHEME WEIGHT SWEEP - {pos}"
          f"   ({active_weeks} weeks where the feature actually moved a row)\n{bar}")
    if active_weeks == 0:
        print("\nEVERY WEEK WAS INERT. The scheme feature did nothing - almost "
              "certainly a missing\nPFF weekly archive for these years "
              "(pff_imports/{year}/weekly/). Nothing here is a result.\n")
        return

    for stat in stats:
        print(f"\n--- {stat} ---")
        print(f"{'weight':<9}{'scope':<12}{'n':>7}{'MAE base':>10}{'MAE var':>10}"
              f"{'dMAE':>9}{'  w-l':>9}{'  95% CI':>20}")
        for w in weights:
            for scope in (pos, f'START-{pos}'):
                b = _block(rows[w][stat].get(scope, []))
                if not b:
                    continue
                print(f"{w:<9.2f}{scope:<12}{b['n']:>7}{b['mae_b']:>10.3f}"
                      f"{b['mae_v']:>10.3f}{b['d']:>+9.3f}"
                      f"{f'  {b['wins']}-{b['losses']}':>9}"
                      f"{f'  [{b['lo']:+.3f},{b['hi']:+.3f}]':>20}"
                      f"{' *' if b['sig'] else ''}")

    seasons = sorted({p[3] for w in weights for s in stats
                      for sc in rows[w][s] for p in rows[w][s][sc]})

    if len(seasons) >= 2:
        # THE test. A weight that is real should hold its sign in each season
        # independently - that is exactly what the coaching cohort tables
        # failed three times running. Per-season columns, not a pooled number,
        # because pooling can hide one season carrying the whole effect.
        print(f"\n\n{bar}\nCROSS-SEASON CHECK - each season scored "
              f"independently (the decisive test)\n{bar}")
        for stat in stats:
            print(f"\n--- {stat} ---")
            hdr = ''.join(f"{str(y) + ' dMAE':>13}" for y in seasons)
            print(f"{'weight':<9}{'scope':<12}{hdr}{'  agree?':>18}")
            for w in weights:
                for scope in (pos, f'START-{pos}'):
                    pairs = rows[w][stat].get(scope, [])
                    per = {y: _block([p for p in pairs if p[3] == y]) for y in seasons}
                    if any(v is None for v in per.values()):
                        continue
                    cells = ''.join(f"{per[y]['d']:>+13.3f}" for y in seasons)
                    signs = {(per[y]['d'] < 0) for y in seasons}
                    agree = 'yes' if len(signs) == 1 else 'NO - sign flips'
                    print(f"{w:<9.2f}{scope:<12}{cells}{agree:>18}")
        print("\n(A weight that flips sign between seasons is not a finding, "
              "whatever the pooled\n number says. Ship only what holds in both.)")
    else:
        # Only one season of archive - the weakest split available.
        print(f"\n\n{bar}\nODD/EVEN WEEK SPLIT (weak within-season check - "
              f"NOT a substitute for a second season)\n{bar}")
        for stat in stats:
            print(f"\n--- {stat} ---")
            print(f"{'weight':<9}{'scope':<12}{'odd dMAE':>11}{'even dMAE':>12}{'  agree?':>10}")
            for w in weights:
                for scope in (pos, f'START-{pos}'):
                    pairs = rows[w][stat].get(scope, [])
                    odd = _block([p for p in pairs if p[2] % 2 == 1])
                    even = _block([p for p in pairs if p[2] % 2 == 0])
                    if not (odd and even):
                        continue
                    agree = 'yes' if (odd['d'] < 0) == (even['d'] < 0) else 'NO - sign flips'
                    print(f"{w:<9.2f}{scope:<12}{odd['d']:>+11.3f}{even['d']:>+12.3f}"
                          f"{agree:>18}")

    print("\n(dMAE negative = the fixed weight beats DEFAULT_FEATURES. "
          "* = 95% CI excludes 0.)")
    if len(seasons) < 2:
        print("ONE SEASON ONLY - exploratory until a second PFF weekly archive "
              "season is backfilled.")


if __name__ == '__main__':
    main()
