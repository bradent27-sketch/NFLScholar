"""Phase A of the alignment/scheme defense-prior study (2026-09-04).

Question from the user: a defense may keep its OVERALL vulnerability to WR/TE
year to year while its SLOT-vs-WIDE and MAN-vs-ZONE splits churn with
personnel and coordinators. So prior-season data might be worth a lot for
the broad matchup and little-or-nothing for the alignment / scheme residual.

This measures that directly, with NO model builds. For each defense, each
season 2022-2025, it builds the FULL-SEASON alignment-defense and
scheme-defense profiles (the same loaders the model uses, as_of_week past
week 18 so the whole regular season is in), then for every consecutive
season pair reports, per channel:

  * r, OLS slope of  value[t] ~ value[t-1]            (raw year-over-year signal)
  * w_prior*  - the weight on last season that minimises MSE predicting this
    season with NO current-season games, shrinking the rest toward neutral
    1.0 (what the model's shrinkage targets) and, separately, toward the
    league mean.
  * PARTIAL carryover - does last season's SPLIT-SPECIFIC deviation
    (slot_ratio - broad_ratio) predict this season's, after this season's
    broad number is already known. This is the number that says whether a
    prior-season slot/scheme residual carries any information the broad
    matchup doesn't.

Channels: broad WR/TE allowed, slot, wide (=non_slot), man, zone - each for
targets / receptions / yards.

    python scripts/analyze_defense_split_transfer.py --years 2022-2025
    python scripts/analyze_defense_split_transfer.py --years 2022-2025 --csv .sweeps/defense_split_transfer.csv
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.pff_alignment import (  # noqa: E402
    load_weekly_alignment_defense_profiles, load_weekly_scheme_defense_profiles)
from data.weekly_projections import load_schedule  # noqa: E402

POSITIONS = ('WR', 'TE')
STATS = ('targets', 'receptions', 'yards')
FULL_SEASON_AS_OF = 25   # past every regular-season week


def _season_ratios(year):
    """{(pos, stat): DataFrame index=defense_team, cols broad/slot/wide/man/zone}."""
    sch = load_schedule(year)
    a = load_weekly_alignment_defense_profiles(year, FULL_SEASON_AS_OF, sch).profiles
    s = load_weekly_scheme_defense_profiles(year, FULL_SEASON_AS_OF, sch).profiles
    out = {}
    for pos in POSITIONS:
        for stat in STATS:
            ap = a[(a['position'] == pos) & (a['stat'] == stat)]
            sp = s[(s['position'] == pos) & (s['stat'] == stat)]
            rows = {}
            for team, g in ap.groupby('defense_team'):
                by = g.set_index('alignment')
                def rr(al):
                    if al not in by.index:
                        return np.nan
                    o, e = by.at[al, 'observed_total'], by.at[al, 'expected_total']
                    return o / e if (pd.notna(e) and e > 0) else np.nan
                slot_o = by.at['slot', 'observed_total'] if 'slot' in by.index else np.nan
                slot_e = by.at['slot', 'expected_total'] if 'slot' in by.index else np.nan
                ns_o = by.at['non_slot', 'observed_total'] if 'non_slot' in by.index else np.nan
                ns_e = by.at['non_slot', 'expected_total'] if 'non_slot' in by.index else np.nan
                broad = ((np.nansum([slot_o, ns_o])) / np.nansum([slot_e, ns_e])
                         if np.nansum([slot_e, ns_e]) > 0 else np.nan)
                rows[team] = dict(broad=broad, slot=rr('slot'), wide=rr('wide'))
            for team, g in sp.groupby('defense_team'):
                by = g.set_index('scheme')
                for sch_name in ('man', 'zone'):
                    if sch_name in by.index:
                        o, e = by.at[sch_name, 'observed_total'], by.at[sch_name, 'expected_total']
                        rows.setdefault(team, {})[sch_name] = o / e if (pd.notna(e) and e > 0) else np.nan
            out[(pos, stat)] = pd.DataFrame(rows).T
    return out


def _fit_w_prior(prev, cur, target):
    """weight w on prev that minimises mean((w*prev + (1-w)*target - cur)^2)."""
    m = np.isfinite(prev) & np.isfinite(cur)
    prev, cur = prev[m], cur[m]
    if len(prev) < 8:
        return np.nan
    tgt = np.full(len(prev), target) if np.isscalar(target) else target[m]
    d = prev - tgt
    denom = float(np.dot(d, d))
    if denom <= 0:
        return np.nan
    return float(np.clip(np.dot(d, cur - tgt) / denom, -0.5, 1.5))


def _rslope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan, np.nan
    x, y = x[m], y[m]
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return r, slope


def _partial(prev_resid, cur_resid):
    """r of last-season split deviation vs this-season split deviation -
    both are (split_ratio - broad_ratio), so this is carryover NET of the
    broad matchup."""
    return _rslope(prev_resid, cur_resid)[0]


def run(years, csv_path):
    per_year = {y: _season_ratios(y) for y in years}
    pairs = list(zip(years[:-1], years[1:]))
    rows = []
    for pos in POSITIONS:
        for stat in STATS:
            print(f"\n{'=' * 96}\n{pos}  {stat}\n{'=' * 96}")
            print(f"{'channel':<10}{'pair':<12}{'n':>5}{'r':>8}{'slope':>8}"
                  f"{'w_prior|neutral':>16}{'w_prior|mean':>14}{'partial r':>11}")
            # pooled across pairs
            pooled = {ch: {'prev': [], 'cur': [], 'presid': [], 'cresid': []}
                      for ch in ('broad', 'slot', 'wide', 'man', 'zone')}
            for y0, y1 in pairs:
                d0 = per_year[y0][(pos, stat)]
                d1 = per_year[y1][(pos, stat)]
                common = d0.index.intersection(d1.index)
                d0, d1 = d0.loc[common], d1.loc[common]
                broad0, broad1 = d0['broad'].to_numpy(float), d1['broad'].to_numpy(float)
                for ch in ('broad', 'slot', 'wide', 'man', 'zone'):
                    if ch not in d0.columns or ch not in d1.columns:
                        continue
                    p, c = d0[ch].to_numpy(float), d1[ch].to_numpy(float)
                    r, slope = _rslope(p, c)
                    w_neu = _fit_w_prior(p, c, 1.0)
                    mean_c = np.nanmean(c)
                    w_mean = _fit_w_prior(p, c, mean_c)
                    presid = p - broad0 if ch != 'broad' else p * np.nan
                    cresid = c - broad1 if ch != 'broad' else c * np.nan
                    part = _partial(presid, cresid) if ch != 'broad' else np.nan
                    n = int((np.isfinite(p) & np.isfinite(c)).sum())
                    print(f"{ch:<10}{f'{y0}->{y1}':<12}{n:>5}{r:>8.3f}{slope:>8.3f}"
                          f"{w_neu:>16.3f}{w_mean:>14.3f}{part:>11.3f}")
                    rows.append(dict(pos=pos, stat=stat, channel=ch, pair=f'{y0}->{y1}', n=n,
                                     r=round(r, 4), slope=round(slope, 4),
                                     w_prior_neutral=round(w_neu, 4), w_prior_mean=round(w_mean, 4),
                                     partial_r=round(part, 4) if np.isfinite(part) else None))
                    pooled[ch]['prev'] += list(p); pooled[ch]['cur'] += list(c)
                    pooled[ch]['presid'] += list(presid); pooled[ch]['cresid'] += list(cresid)
            print(f"{'-' * 96}")
            for ch, d in pooled.items():
                if not d['prev']:
                    continue
                p, c = np.array(d['prev']), np.array(d['cur'])
                pr, cr = np.array(d['presid']), np.array(d['cresid'])
                r, slope = _rslope(p, c)
                w_neu = _fit_w_prior(p, c, 1.0)
                w_mean = _fit_w_prior(p, c, np.nanmean(c))
                part = _partial(pr, cr) if ch != 'broad' else np.nan
                n = int((np.isfinite(p) & np.isfinite(c)).sum())
                print(f"{ch:<10}{'POOLED':<12}{n:>5}{r:>8.3f}{slope:>8.3f}"
                      f"{w_neu:>16.3f}{w_mean:>14.3f}{part:>11.3f}")
                rows.append(dict(pos=pos, stat=stat, channel=ch, pair='POOLED', n=n,
                                 r=round(r, 4), slope=round(slope, 4),
                                 w_prior_neutral=round(w_neu, 4), w_prior_mean=round(w_mean, 4),
                                 partial_r=round(part, 4) if np.isfinite(part) else None))

    print(f"\n{'=' * 96}\nREAD\n{'=' * 96}")
    print("w_prior|neutral is how hard to weight LAST season (rest toward 1.0) with NO games in.\n"
          "A big gap between 'broad' and 'slot'/'man' w_prior, and a near-zero 'partial r', is the\n"
          "user's hypothesis confirmed: use the prior for the broad matchup, not for the split.\n"
          "A healthy 'partial r' on slot/man means the split itself carries year-to-year signal\n"
          "and a prior-season residual is worth blending in too.")
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nwrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2022-2025')
    ap.add_argument('--csv', default='.sweeps/defense_split_transfer.csv')
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split('-'))
    run(list(range(y0, y1 + 1)), a.csv)


if __name__ == '__main__':
    main()
