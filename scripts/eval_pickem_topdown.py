"""
B5 study (2026-11-11, docs/audit_leads_2026-11-10.md): pick'em legs picked from the sharp price alone (data/pickem.py), scored on
every archived week with results. The line and the sharp price are the last pulls before the player's kickoff. The model's side
(from the prop record's 'at close' rows, .sweeps/prop_scorecard_rows.csv, written by scripts/prop_scorecard.py) is a column, not
a filter.

    python scripts/eval_pickem_topdown.py [--year 2026] [--out .sweeps/pickem_topdown.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import pandas as pd  # noqa: E402

from data import pickem  # noqa: E402
from data.ledger_scorecard import kickoffs_utc  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--out', default=os.path.join('.sweeps', 'pickem_topdown.txt'))
    ap.add_argument('--record-rows', default=os.path.join('.sweeps', 'prop_scorecard_rows.csv'))
    a = ap.parse_args()
    stats, team_col, name_col, _ = load_and_merge_data(a.year, 'Full PPR')
    stats = stats[stats['season'] == a.year] if 'season' in stats.columns else stats
    actual = stats[['week', name_col, team_col] + [s for s in pickem.MARKET_STAT.values() if s in stats.columns]].copy()
    actual['week'] = actual['week'].astype(int)
    actual['_key'] = clean_name_exact(actual[name_col].astype(str))
    ko = kickoffs_utc(load_schedule(a.year))
    cutoff = {(int(w), k): ko.get((int(w), str(t))) for w, k, t in zip(actual['week'], actual['_key'], actual[team_col])}
    cutoff = {k: v for k, v in cutoff.items() if v is not None}
    lines = pickem.load_lines(a.year)
    legs = pickem.choose_sides(pickem.legs_at(lines, cutoff))
    if os.path.exists(a.record_rows):
        rec = pd.read_csv(a.record_rows)
        rec = rec[(rec['timing'] == 'at close') & rec['sharpest'].astype(bool) & rec['p_over'].notna()]
        legs = pickem.model_agrees(legs, rec)
    scored = pickem.attach_results(legs, actual.drop(columns=[name_col, team_col]))
    be = pickem.break_evens()
    pay = pickem.load_payouts()
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    weeks = sorted(int(w) for w in scored['week'].unique())
    head = [f"{a.year} pick'em legs from the sharp price (B5), weeks {weeks}: {len(scored):,} lines with a result and a sharp "
            "price, at the last pulls before kickoff. Flagged = the fair chance (the sharp book's de-vigged price moved to the "
            "pick'em line) clears the leg's break-even.",
            'Break-even per leg: ' + '; '.join(f"{b} {v[0]:.3f} ({v[1]}; 2-pick {v[2]:.3f}; "
                                               f"{'verified' if pay[b].get('verified') else 'UNVERIFIED payouts'})"
                                               for b, v in be.items()),
            f"Bar: surface a product and stat only when flagged legs beat their break-even with a 90% lower bound above it on "
            f"{pickem.SURFACE_MIN_LEGS}+ legs. 'model agree' = the model's P at the line is on the leg's side.", '']
    body = ['1. BY PRODUCT', pickem.leg_table(scored, ('book', 'kind')).to_string(index=False, float_format=fmt), '',
            '2. BY PRODUCT AND STAT', pickem.leg_table(scored, ('book', 'kind', 'stat')).to_string(index=False, float_format=fmt),
            '', '3. BY PRODUCT AND SIDE', pickem.leg_table(scored, ('book', 'kind', 'side')).to_string(index=False, float_format=fmt),
            '', '4. BY SHARP BOOK', pickem.leg_table(scored, ('book', 'sharp_book')).to_string(index=False, float_format=fmt)]
    text = '\n'.join(head + body)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    scored.to_csv(a.out.replace('.txt', '_legs.csv'), index=False)


if __name__ == '__main__':
    main()
