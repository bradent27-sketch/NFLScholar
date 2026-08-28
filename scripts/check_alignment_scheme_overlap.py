"""
Checks how much the slot/wide/inline alignment candidate and the man/zone
scheme candidate overlap - i.e. whether combining them risks crediting the
same real defensive vulnerability twice under two different names.

Built 2026-08-27 per explicit request, ahead of any decision about how (or
whether) to combine v2_pff_alignment_matchup and v2_scheme_matchup: before
picking a blend weight, first find out whether the two signals are largely
redundant (high correlation -> treat as substitutes, not additive) or mostly
independent (low correlation -> safe to combine).

Both candidate_multiplier columns are already computed and exposed in the
decomposition whenever 'v2_pff_alignment_matchup' is in feats (DEFAULT_
FEATURES always has it on) - this needs no new flag and does not touch
scoring at all, it only reads what the shipped model already calculates and
discards after display.

Usage:
    python scripts/check_alignment_scheme_overlap.py --years 2024,2025 --weeks 2-18
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.weekly_projections import build_weekly_projections, DEFAULT_FEATURES  # noqa: E402

STATS = ('targets', 'receptions', 'yards')
POSITIONS = ('WR', 'TE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2024,2025')
    ap.add_argument('--weeks', default='2-18')
    ap.add_argument('--scoring', default='Full PPR')
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(',')]
    if '-' in args.weeks:
        lo, hi = args.weeks.split('-')
        weeks = list(range(int(lo), int(hi) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(',')]

    records = []
    for year in years:
        for week in weeks:
            proj, meta = build_weekly_projections(
                year, week, args.scoring, as_of_week=week, apply_injury=False, features=DEFAULT_FEATURES)
            if proj.empty:
                continue
            for pos in POSITIONS:
                sub = proj[proj['Pos'] == pos]
                for stat in STATS:
                    align_col = f'_profile_alignment_defense_{stat}_candidate_multiplier'
                    scheme_col = f'_scheme_profile_scheme_defense_{stat}_candidate_multiplier'
                    align_avail_col = '_profile_alignment_defense_candidate_available'
                    scheme_avail_col = '_scheme_profile_scheme_defense_candidate_available'
                    if not all(c in sub.columns for c in (align_col, scheme_col, align_avail_col, scheme_avail_col)):
                        continue
                    for _, r in sub.iterrows():
                        records.append({
                            'year': year, 'week': week, 'pos': pos, 'stat': stat,
                            'player': r['Player'],
                            'align_mult': r.get(align_col), 'scheme_mult': r.get(scheme_col),
                            'align_avail': bool(r.get(align_avail_col, False)),
                            'scheme_avail': bool(r.get(scheme_avail_col, False)),
                        })

    df = pd.DataFrame(records)
    if df.empty:
        print("no data collected")
        return

    print(f"total rows scored: {len(df)}")
    for pos in POSITIONS:
        for stat in STATS:
            sub = df[(df['pos'] == pos) & (df['stat'] == stat)]
            if sub.empty:
                continue
            both = sub[sub['align_avail'] & sub['scheme_avail']]
            align_only = sub[sub['align_avail'] & ~sub['scheme_avail']]
            scheme_only = sub[~sub['align_avail'] & sub['scheme_avail']]
            neither = sub[~sub['align_avail'] & ~sub['scheme_avail']]
            print(f"\n-- {pos} {stat} (n={len(sub)}) --")
            print(f"  both available:    {len(both):>5} ({100*len(both)/len(sub):.1f}%)")
            print(f"  alignment only:    {len(align_only):>5} ({100*len(align_only)/len(sub):.1f}%)")
            print(f"  scheme only:       {len(scheme_only):>5} ({100*len(scheme_only)/len(sub):.1f}%)")
            print(f"  neither:           {len(neither):>5} ({100*len(neither)/len(sub):.1f}%)")
            if len(both) >= 20:
                a = pd.to_numeric(both['align_mult'], errors='coerce')
                s = pd.to_numeric(both['scheme_mult'], errors='coerce')
                ok = a.notna() & s.notna()
                if ok.sum() >= 20:
                    r = a[ok].corr(s[ok])
                    # Residual (deviation from 1.0) correlation is the more
                    # relevant number - the multipliers both center near 1.0,
                    # so correlating the raw values is dominated by that
                    # shared center; the residual isolates whether the
                    # DIRECTION AND SIZE of each one's departure from neutral
                    # agree.
                    a_resid = a[ok] - 1.0
                    s_resid = s[ok] - 1.0
                    r_resid = a_resid.corr(s_resid)
                    print(f"  r(raw multipliers, n={int(ok.sum())}):      {r:+.3f}")
                    print(f"  r(residuals from 1.0):        {r_resid:+.3f}")
                    print(f"  mean |align resid|: {a_resid.abs().mean():.3f}   mean |scheme resid|: {s_resid.abs().mean():.3f}")


if __name__ == '__main__':
    main()
