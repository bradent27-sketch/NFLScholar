"""Phase B of the alignment/scheme defense-prior study (2026-09-04).

Sweeps the per-channel prior-season blend weight w0 for
'v2_pff_defense_prior_blend' (data.pff_alignment.PFF_DEFENSE_PRIOR_BLEND_W0)
on EARLY-SEASON weeks, where a completed prior season is the most evidence a
defense residual has. The transfer study (Phase A) predicts scheme lands
near w0~0.4 and alignment near 0; this checks that against held-out weekly
MAE.

    python scripts/sweep_defense_prior_blend.py --years 2023,2024,2025 --weeks 1-8
    python scripts/sweep_defense_prior_blend.py --years 2023,2024,2025 --weeks 3-18 --scheme-only 0.4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from data.weekly_projections import DEFAULT_FEATURES  # noqa: E402
from scripts._align_scheme_eval import evaluate  # noqa: E402

FLAG = 'v2_pff_defense_prior_blend'
_ORIG = {'alignment': 0.0, 'scheme': 0.40}


def _mutator(align_w, scheme_w):
    def m(wp, reset=False):
        d = wp.PFF_DEFENSE_PRIOR_BLEND_W0
        if reset:
            d['alignment'], d['scheme'] = _ORIG['alignment'], _ORIG['scheme']
        else:
            d['alignment'], d['scheme'] = align_w, scheme_w
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2023,2024,2025')
    ap.add_argument('--weeks', default='1-8')
    ap.add_argument('--scheme-only', type=float, default=None,
                    help="run a single variant: scheme w0 = this, alignment w0 = 0")
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    weeks = list(range(w0, w1 + 1))
    feats = frozenset(DEFAULT_FEATURES | {FLAG})

    if a.scheme_only is not None:
        variants = [(f"scheme w0={a.scheme_only:.2f}, align 0", feats, _mutator(0.0, a.scheme_only))]
    else:
        variants = []
        for sw in (0.20, 0.30, 0.40, 0.50, 0.60):
            variants.append((f"scheme w0={sw:.2f} / align 0.00", feats, _mutator(0.0, sw)))
        for aw in (0.20, 0.40):
            variants.append((f"scheme w0=0.40 / align w0={aw:.2f}", feats, _mutator(aw, 0.40)))

    evaluate(variants, years, weeks,
             outliers_csv='.sweeps/defense_prior_blend_outliers.csv')


if __name__ == '__main__':
    main()
