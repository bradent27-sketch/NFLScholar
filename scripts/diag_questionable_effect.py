"""Do Questionable players who play produce less than the model projects? (2026-10-08)

Reads the calibration-lab dump (the harness base arm: the shipped 'final' projection beside the actual, 2021-2025)
and joins the official nflverse injury report (`data/historical_availability.load_injury_reports`) to it, to answer:

  1. WHEN in the week is the Questionable tag filed? (the backtest sees each player's last report before kickoff)
  2. How big is the shortfall, per position and season, with a week-clustered bootstrap CI, controlling for the
     projection level (a Q player is not the same kind of player as the average starter)?
  3. Does the report's own practice status (Full / Limited / DNP) explain it? Is it the tag or the practice?
  4. Is it a volume shortfall (targets, carries) or an efficiency one, and how much is touchdowns (noisy)?

    python scripts/diag_questionable_effect.py --dump-path .sweeps/calibration_lab_v7.csv --week1-path .sweeps/calibration_lab_v7_wk1_ourlads.csv
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

import scripts.eval_calibration_schemes as ecs  # noqa: E402
from data.historical_availability import load_injury_reports, _kickoff_by_team  # noqa: E402
from data.loaders import load_schedule  # noqa: E402

POS = ('QB', 'RB', 'WR', 'TE')


def attach_reports(d):
    """Join each dump row to its official report row (status, practice status, filing time vs kickoff)."""
    out = []
    for year in sorted(d['year'].unique()):
        rep = load_injury_reports(int(year))
        sched = load_schedule(int(year))
        rep = rep.copy()
        rep['week'] = pd.to_numeric(rep['week'], errors='coerce')
        # 2025 and 2026 reports in the nflverse cache carry no filing timestamp: timing columns stay NaN there
        rep['modified'] = (pd.to_datetime(rep['date_modified'], errors='coerce', utc=True)
                           if 'date_modified' in rep.columns else pd.NaT)
        rep = rep.sort_values('modified').drop_duplicates(['week', 'gsis_id'], keep='last')
        ko = {}
        for wk in sorted(rep['week'].dropna().unique()):
            ko[int(wk)] = _kickoff_by_team(sched[pd.to_numeric(sched['week'], errors='coerce') == wk])
        rep['kickoff'] = [ko.get(int(w), {}).get(str(t).upper()) if pd.notna(w) else None
                          for w, t in zip(rep['week'], rep['team'])]
        rep['kickoff'] = pd.to_datetime(rep['kickoff'], errors='coerce', utc=True)
        rep['modified'] = pd.to_datetime(rep['modified'], errors='coerce', utc=True)
        rep['hours_before'] = (rep['kickoff'] - rep['modified']).dt.total_seconds() / 3600.0
        rep['dow_et'] = rep['modified'].dt.tz_convert('US/Eastern').dt.day_name()
        rep['key'] = rep['full_name'].astype(str) + '|' + rep['week'].astype('Int64').astype(str)
        sub = d[d['year'] == year].copy()
        sub['key'] = sub['player'].astype(str) + '|' + sub['week'].astype(str)
        rep = rep.drop_duplicates('key')
        sub = sub.merge(rep[['key', 'report_status', 'practice_status', 'hours_before', 'dow_et']], on='key', how='left')
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def boot_ratio(x, n_boot=2000, seed=0):
    """actual-sum / projected-sum over rows, with a bootstrap CI over (year, week) clusters."""
    rng = np.random.default_rng(seed)
    g = x.groupby(['year', 'week']).agg(a=('actual', 'sum'), p=('final', 'sum'))
    a, p = g['a'].to_numpy(), g['p'].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    r = a[idx].sum(1) / p[idx].sum(1)
    return a.sum() / p.sum(), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-path', default=os.path.join('.sweeps', 'calibration_lab_v7.csv'))
    ap.add_argument('--week1-path', default='')
    ap.add_argument('--weeks', default='3-17')
    a = ap.parse_args()
    d = ecs.load_dump(a.dump_path, a.week1_path)
    w0, w1 = (int(x) for x in a.weeks.split('-'))
    d = d[d['week'].between(w0, w1)].copy()
    d = attach_reports(d)
    d['start'] = ecs.top_n_mask(d, d['final'].to_numpy(dtype=float))
    d['tag'] = d['report_status'].fillna('none').astype(str)
    live = d[(d['avail'] > 0.01) & (d['final'] > 0.005)]
    pd.set_option('display.width', 220)

    # ---- 1. timing ----
    q = live[live['tag'] == 'Questionable']
    print('=' * 100 + f"\n1. WHEN is the Questionable tag filed? (last report before kickoff; weeks {a.weeks}, n={len(q):,} Q rows)\n" + '=' * 100)
    print(q['dow_et'].value_counts(normalize=True).round(3).to_string())
    hb = q['hours_before'].dropna()
    print(f"hours before kickoff: median {hb.median():.0f}, p10 {hb.quantile(.1):.0f}, p90 {hb.quantile(.9):.0f}; "
          f"share filed < 48h before kickoff {(hb < 48).mean():.2f}, < 24h {(hb < 24).mean():.2f}, > 72h {(hb > 72).mean():.2f}")
    print(f"Q rows missing a report join (name mismatch): {q['hours_before'].isna().mean():.3f}")

    # ---- 2. size, per position / pool / season ----
    print('\n' + '=' * 100 + "\n2. actual / projection when he plays (shipped final), Q vs healthy-tagged, played rows\n" + '=' * 100)
    rows = []
    for pos in POS + ('ALL',):
        for pool_name, pool in (('START', live['start']), ('ALL', pd.Series(True, index=live.index))):
            x = live[(live['pos'] == pos) | (pos == 'ALL')]
            x = x[pool.loc[x.index] & x['played']]
            for lab, sub in (('Questionable', x[x['tag'] == 'Questionable']), ('Doubtful', x[x['tag'] == 'Doubtful']),
                             ('none', x[x['tag'] == 'none'])):
                if len(sub) < 25:
                    continue
                r, lo, hi = boot_ratio(sub)
                yrs = sub.groupby('year').apply(lambda s: s['actual'].sum() / s['final'].sum(), include_groups=False)
                rows.append({'pos': pos, 'pool': pool_name, 'tag': lab, 'n': len(sub),
                             'proj': round(sub['final'].mean(), 2), 'actual': round(sub['actual'].mean(), 2),
                             'act/proj': round(r, 3), 'CI': f"[{lo:.3f},{hi:.3f}]",
                             'by season': ' '.join(f"{v:.2f}" for v in yrs.to_numpy())})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 2b. controlling for projection level: actual = a + b*proj + c*Q, per position (START pool + ALL) ----
    print('\n' + '=' * 100 + "\n2b. regression actual ~ a + b*projection + c*[Questionable], played rows (c in points; cluster-bootstrap over weeks)\n" + '=' * 100)
    rows = []
    for pos in POS:
        for pool_name in ('START', 'ALL'):
            x = live[(live['pos'] == pos) & live['played'] & live['tag'].isin(['Questionable', 'none'])]
            if pool_name == 'START':
                x = x[x['start']]
            X = np.column_stack([np.ones(len(x)), x['final'].to_numpy(), (x['tag'] == 'Questionable').to_numpy().astype(float)])
            y = x['actual'].to_numpy()
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            keys = (x['year'].astype(str) + '-' + x['week'].astype(str)).to_numpy()
            uniq = np.unique(keys)
            rng = np.random.default_rng(1)
            bs = []
            for _ in range(600):
                pick = rng.choice(uniq, size=len(uniq))
                ii = np.concatenate([np.flatnonzero(keys == k) for k in pick])
                bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0][2])
            rows.append({'pos': pos, 'pool': pool_name, 'n': len(x), 'nQ': int(X[:, 2].sum()), 'slope on proj': round(beta[1], 3),
                         'Q effect (pts)': round(beta[2], 2), 'CI': f"[{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]"})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- 3. practice status ----
    print('\n' + '=' * 100 + "\n3. by the report's own practice status (the same row that carries the tag), START pool, played\n" + '=' * 100)
    d['prac'] = d['practice_status'].fillna('none').astype(str).str.replace(r' in Practice| Participation| Participate', '', regex=True)
    rows = []
    for pos in ('WR', 'RB', 'TE', 'QB', 'ALL'):
        x = live.assign(prac=d.loc[live.index, 'prac'])
        x = x[(x['pos'] == pos) | (pos == 'ALL')]
        x = x[x['start'] & x['played']]
        for tag in ('Questionable', 'none'):
            for prac, sub in x[x['tag'] == tag].groupby('prac'):
                if len(sub) < 25:
                    continue
                r, lo, hi = boot_ratio(sub)
                rows.append({'pos': pos, 'tag': tag, 'practice': prac, 'n': len(sub), 'act/proj': round(r, 3), 'CI': f"[{lo:.3f},{hi:.3f}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    # also: share of Q rows that play, by practice
    x = live.assign(prac=d.loc[live.index, 'prac'])
    x = x[(x['tag'] == 'Questionable') & x['start']]
    print('\nP(plays | Questionable, START pool) by practice status:')
    print(x.groupby('prac').agg(n=('played', 'size'), played=('played', 'mean')).round(3).to_string())

    # ---- 4. which stat ----
    print('\n' + '=' * 100 + "\n4. which stat: actual / projected per stat, START pool, played, Q vs none\n" + '=' * 100)
    rows = []
    stats = {'WR': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
             'RB': ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receiving_yards'),
             'TE': ('targets', 'receptions', 'receiving_yards', 'receiving_tds'),
             'QB': ('passing_attempts', 'passing_yards', 'passing_tds', 'rushing_yards')}
    for pos, cols in stats.items():
        x = live[(live['pos'] == pos) & live['start'] & live['played']]
        for s in cols:
            if f'p_{s}' not in x or f'a_{s}' not in x:
                continue
            r = {'pos': pos, 'stat': s}
            for tag in ('Questionable', 'none'):
                sub = x[x['tag'] == tag]
                r[tag] = round(sub[f'a_{s}'].sum() / max(sub[f'p_{s}'].sum(), 1e-9), 3)
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
