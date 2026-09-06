"""Phase C of the alignment/scheme study (2026-09-04).

Now that 2022 and 2023 have weekly PFF archives (reorg_pff_weekly_archive.py,
2026-09-04), the alignment/scheme mechanisms can be re-checked on FOUR
seasons instead of the two they were last judged on. Four questions:

  1. ABLATE v2_pff_alignment_matchup  - does the WR/TE slot/wide residual
     still earn its place in DEFAULT_FEATURES on 4x the data?
  2. ABLATE v2_scheme_matchup         - does the TE man/zone residual?
  3. WR SCHEME SCORING - add WR to SCHEME_MATCHUP_SCORING_POSITIONS (today
     TE-only). Two seasons couldn't show a WR scheme win; can four?
  4. BLEND WEIGHT - v2_scheme_alignment_blend with a fixed scheme share
     swept 0 / .25 / .5 / .75 / 1, WR and TE separately (the "lost result"
     from docs/weekly_rankings_backlog.md).

    python scripts/reconfirm_alignment_scheme.py --mode ablate --years 2022,2023,2024,2025 --weeks 3-18
    python scripts/reconfirm_alignment_scheme.py --mode wr_scheme --years 2022,2023,2024,2025 --weeks 3-18
    python scripts/reconfirm_alignment_scheme.py --mode blend --years 2022,2023,2024,2025 --weeks 3-18
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


def _wr_scheme_mutator():
    orig = None

    def m(wp, reset=False):
        nonlocal orig
        if reset:
            if orig is not None:
                wp.SCHEME_MATCHUP_SCORING_POSITIONS = orig
        else:
            orig = wp.SCHEME_MATCHUP_SCORING_POSITIONS
            wp.SCHEME_MATCHUP_SCORING_POSITIONS = frozenset({'TE', 'WR'})
    return m


def _blend_mutator(pos, w):
    def m(wp, reset=False):
        wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT.clear()
        if not reset:
            wp.SCHEME_ALIGNMENT_BLEND_FIXED_WEIGHT[pos] = w
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['ablate', 'wr_scheme', 'blend'])
    ap.add_argument('--years', default='2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-18')
    a = ap.parse_args()
    years = [int(x) for x in a.years.replace(' ', '').split(',')]
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    weeks = list(range(w0, w1 + 1))

    if a.mode == 'ablate':
        variants = [
            ('ablate v2_pff_alignment_matchup', frozenset(DEFAULT_FEATURES - {'v2_pff_alignment_matchup'}), None),
            ('ablate v2_scheme_matchup', frozenset(DEFAULT_FEATURES - {'v2_scheme_matchup'}), None),
        ]
        evaluate(variants, years, weeks, outliers_csv='.sweeps/alignment_scheme_ablate_outliers.csv')
    elif a.mode == 'wr_scheme':
        variants = [('WR added to scheme scoring', DEFAULT_FEATURES, _wr_scheme_mutator())]
        evaluate(variants, years, weeks, outliers_csv='.sweeps/wr_scheme_scoring_outliers.csv')
    else:
        feats = frozenset(DEFAULT_FEATURES | {'v2_scheme_alignment_blend'})
        variants = []
        for pos in ('WR', 'TE'):
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                variants.append((f"blend {pos} scheme-share={w:.2f}", feats, _blend_mutator(pos, w)))
        # Rides along at low marginal cost (one more build/week, same years/
        # weeks already being built here): does the ACTUALLY-SHIPPED
        # v2_scheme_matchup replace-mechanism itself hold up on
        # receiving_yards per season, not just pooled? The old fixed-weight
        # blend sweep (.sweeps/scheme_blend_TE_v2.txt) found weight=1.0
        # (closest analogue to "replace") sign-flips on START-TE
        # receiving_yards between 2024 (-0.510) and 2025 (+0.097) - but that
        # blend formula drops to neutral (not alignment) when scheme lacks
        # evidence, unlike the real replace mechanism's alignment fallback,
        # so it was never a direct test of what's shipped. This is.
        variants.append(('ablate v2_scheme_matchup (shipped, TE)',
                         frozenset(DEFAULT_FEATURES - {'v2_scheme_matchup'}), None))
        evaluate(variants, years, weeks)


if __name__ == '__main__':
    main()
