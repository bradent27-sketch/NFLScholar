"""Shared paired-A/B evaluation for the 2026-09-04 alignment/scheme study.

Both scripts that use it (sweep_defense_prior_blend.py, reconfirm_alignment_
scheme.py) need the same thing: build a DEFAULT board and one or more variant
boards per week over a window, score each on the intersection player pool,
and report scope MAE (ALL / per-pos / startable) with a bootstrap CI + sign
test, plus a per-stat MAE readout over the startable receiving pool.

A variant is either a different feature set, or DEFAULT_FEATURES with a
monkeypatch applied first (SCHEME_MATCHUP_SCORING_POSITIONS, a blend-weight
dict, PFF_DEFENSE_PRIOR_BLEND_W0 - all read live at build time, so a
build_weekly_projections.clear() between variants is mandatory since the
cache key is `features`, which does not change when only a constant moves).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from scripts.eval_weekly_model import _metrics, _weighted, STARTABLE_N, _actual_points  # noqa: E402

SCOPES = [('ALL', None, False), ('WR', 'WR', False), ('TE', 'TE', False),
          ('START-WR', 'WR', True), ('START-TE', 'TE', True),
          ('START-QB', 'QB', True), ('START-RB', 'RB', True)]
PER_STAT = ('targets', 'receptions', 'receiving_yards')


def _scope_df(df, pos, startable):
    d = df if pos is None else df[df['Pos'] == pos]
    if startable:
        d = d.nlargest(STARTABLE_N.get(pos, 30), 'Model Proj Pts')
    return d


def boot_ci(deltas, weights, n=3000, seed=0):
    if len(deltas) < 4:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float); w = np.asarray(weights, float)
    idx = np.arange(len(d)); out = np.empty(n)
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        out[b] = np.average(d[s], weights=w[s])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return float('nan')
    k = max(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def _stat_actuals(stats_df, name_col, week):
    rows = stats_df[pd.to_numeric(stats_df['week'], errors='coerce') == week]
    present = [s for s in PER_STAT if s in rows.columns]
    return rows.groupby(name_col, observed=True)[present].sum()


def evaluate(variants, years, weeks, scoring='Full PPR', outliers_csv=None):
    """`variants` = list of (label, feature_set, mutate_fn_or_None). Each
    mutate_fn takes the `wp` module and sets whatever constant it needs;
    it is CALLED before the variant build and UNDONE (call with None) by
    restoring from a saved snapshot the caller manages - simplest is for
    mutate_fn to fully set the value each call and a final reset_fn to
    restore. Here we just clear the cache and re-apply per week."""
    scoring_col = 'fantasy_points_ppr' if scoring != 'Standard' else 'fantasy_points'
    scope_pairs = {lbl: {s[0]: [] for s in SCOPES} for lbl, _f, _m in variants}
    stat_pairs = {lbl: {s: [] for s in PER_STAT} for lbl, _f, _m in variants}
    # Same accumulation, split by year - a weight that only wins pooled and
    # flips sign scored per-season is not a finding (see the 2026-09-02
    # scheme-blend sweep: TE receiving_yards at weight=1.0 was -0.510 in 2024
    # but +0.097 in 2025). _report checks agreement across every year here.
    stat_pairs_by_year = {lbl: {yr: {s: [] for s in PER_STAT} for yr in years} for lbl, _f, _m in variants}
    scope_pairs_by_year = {lbl: {yr: {s[0]: [] for s in SCOPES} for yr in years} for lbl, _f, _m in variants}
    outliers = []

    for year in years:
        stats_df, _tc, name_col, _ = load_and_merge_data(year, scoring)
        if 'week' not in stats_df.columns:
            print(f"{year}: no weekly data", flush=True)
            continue
        for week in weeks:
            actual = _actual_points(stats_df, name_col, week, scoring_col)
            if actual.empty:
                continue
            build_weekly_projections.clear()
            base, bmeta = build_weekly_projections(
                year, week, scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if base.empty:
                print(f"{year} w{week}: base empty ({bmeta.get('reason')})")
                continue
            sa = _stat_actuals(stats_df, name_col, week)

            for lbl, feats, mutate in variants:
                if mutate:
                    mutate(wp)
                build_weekly_projections.clear()
                var, vmeta = build_weekly_projections(
                    year, week, scoring, as_of_week=week, apply_injury=False, features=feats)
                if mutate:
                    mutate(wp, reset=True)
                    build_weekly_projections.clear()
                if var.empty:
                    continue
                pool = sorted(set(base['Player']) & set(var['Player']))
                if len(pool) < 20:
                    continue
                b = base[base['Player'].isin(pool)]; v = var[var['Player'].isin(pool)]
                for scope, pos, startable in SCOPES:
                    mb = _metrics(pd.Series(_scope_df(b, pos, startable)['Model Proj Pts'].to_numpy(),
                                            index=_scope_df(b, pos, startable)['Player']), actual)
                    mv = _metrics(pd.Series(_scope_df(v, pos, startable)['Model Proj Pts'].to_numpy(),
                                            index=_scope_df(v, pos, startable)['Player']), actual)
                    if mb and mv:
                        scope_pairs[lbl][scope].append((mb, mv))
                        scope_pairs_by_year[lbl][year][scope].append((mb, mv))
                for stat in PER_STAT:
                    if stat not in b.columns or stat not in sa.columns:
                        continue
                    bd = pd.concat([_scope_df(b, p, True) for p in ('WR', 'TE', 'RB')]).set_index('Player')
                    vd = v.set_index('Player')
                    common = bd.index.intersection(vd.index).intersection(sa.index)
                    if len(common) < 5:
                        continue
                    act = pd.to_numeric(sa.loc[common, stat], errors='coerce')
                    eb = (pd.to_numeric(bd.loc[common, stat], errors='coerce') - act).abs().mean()
                    ev = (pd.to_numeric(vd.loc[common, stat], errors='coerce') - act).abs().mean()
                    if np.isfinite(eb) and np.isfinite(ev):
                        stat_pairs[lbl][stat].append((float(eb), float(ev), len(common)))
                        stat_pairs_by_year[lbl][year][stat].append((float(eb), float(ev), len(common)))
                if outliers_csv:
                    for pos in ('WR', 'TE'):
                        vd = _scope_df(v, pos, True).set_index('Player')
                        bd = b.set_index('Player')
                        for player, r in vd.iterrows():
                            if player not in actual.index:
                                continue
                            pv, av = float(r['Model Proj Pts']), float(actual[player])
                            bp = float(bd.loc[player, 'Model Proj Pts']) if player in bd.index else np.nan
                            outliers.append(dict(label=lbl, year=year, week=week, player=player, pos=pos,
                                                 base=round(bp, 2), var=round(pv, 2), actual=round(av, 2),
                                                 shift=round(pv - bp, 3), abs_error=round(abs(pv - av), 2)))
            print(f"{year} w{week}: pool ok", flush=True)

    _report(variants, scope_pairs, stat_pairs, years, weeks, scope_pairs_by_year, stat_pairs_by_year)
    if outliers_csv and outliers:
        os.makedirs(os.path.dirname(outliers_csv) or '.', exist_ok=True)
        pd.DataFrame(outliers).sort_values('abs_error', ascending=False).to_csv(outliers_csv, index=False)
        print(f"\nwrote {outliers_csv}")
    return scope_pairs, stat_pairs


def _year_dmae(pairs):
    """Weighted dMAE for one (year, scope-or-stat) cell, or None if empty.
    Handles both scope pairs (mb, mv) dicts and stat pairs (eb, ev, n) tuples."""
    if not pairs:
        return None
    if isinstance(pairs[0][0], dict):
        mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
        return _weighted(mv, 'mae') - _weighted(mb, 'mae')
    eb = np.array([r[0] for r in pairs]); ev = np.array([r[1] for r in pairs])
    w = np.array([r[2] for r in pairs], float)
    return float(np.average(ev, weights=w)) - float(np.average(eb, weights=w))


def _report(variants, scope_pairs, stat_pairs, years, weeks, scope_pairs_by_year=None, stat_pairs_by_year=None):
    bar = '=' * 96
    print(f"\n{bar}\nALIGNMENT/SCHEME A/B   years={years} weeks={weeks}\n{bar}")
    print("dMAE = variant - base.  NEGATIVE = variant more accurate.  * = bootstrap 95% CI excludes 0.\n")
    for lbl, _f, _m in variants:
        print(f"--- {lbl}")
        print(f"{'scope':<12}{'n':>8}{'MAEbase':>10}{'MAEvar':>10}{'dMAE':>9}{'95% CI':>20}{'W-L':>9}{'p':>8}")
        for scope, _p, _s in SCOPES:
            pairs = scope_pairs[lbl].get(scope) or []
            if not pairs:
                continue
            mb = [p[0] for p in pairs]; mv = [p[1] for p in pairs]
            nt = sum(m['n'] for m in mb)
            base_mae, var_mae = _weighted(mb, 'mae'), _weighted(mv, 'mae')
            deltas = [b['mae'] - a['mae'] for a, b in zip(mb, mv)]
            wts = [a['n'] for a in mb]
            lo, hi = boot_ci(deltas, wts)
            w = sum(1 for d in deltas if d < 0); l = sum(1 for d in deltas if d > 0)
            star = ' *' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else '  '
            print(f"{scope:<12}{nt:>8}{base_mae:>10.3f}{var_mae:>10.3f}{var_mae - base_mae:>+9.3f}"
                  f"  [{lo:+.3f},{hi:+.3f}]{star}{f'{w}-{l}':>9}{sign_p(w, l):>8.3f}")
        print(f"{'per-stat (startable recv pool)':<30}")
        for stat in PER_STAT:
            rows = stat_pairs[lbl].get(stat) or []
            if not rows:
                continue
            eb = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
            ww = np.array([r[2] for r in rows], float)
            bm = float(np.average(eb, weights=ww)); vm = float(np.average(ev, weights=ww))
            lo, hi = boot_ci(list(ev - eb), list(ww))
            star = ' *' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else '  '
            print(f"  {stat:<20}{int(ww.sum()):>8}{bm:>10.3f}{vm:>10.3f}{vm - bm:>+9.3f}  [{lo:+.3f},{hi:+.3f}]{star}")

        if stat_pairs_by_year and len(years) > 1:
            print(f"  per-season stability (startable recv pool) - each year scored independently:")
            for stat in PER_STAT:
                cells = [(yr, _year_dmae(stat_pairs_by_year[lbl][yr].get(stat) or [])) for yr in years]
                cells = [(yr, d) for yr, d in cells if d is not None]
                if len(cells) < 2:
                    continue
                signs = {1 if d > 0 else (-1 if d < 0 else 0) for _, d in cells}
                agree = 'yes' if len(signs - {0}) <= 1 else 'NO - sign flips'
                by_year = '  '.join(f"{yr}:{d:+.3f}" for yr, d in cells)
                print(f"    {stat:<18}{by_year:<40}  agree? {agree}")
            print(f"  (a stat whose sign flips year to year is not a finding, whatever the pooled row says)")
        print()
