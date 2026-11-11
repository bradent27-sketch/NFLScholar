"""
A1 screen (2026-11-11): which Questionable rule predicts production best, held out by season? Reads the panel
scripts/study_questionable_timing.py writes (.sweeps/study_questionable_timing_panel.parquet: played rows, 2022-2026).

Every rule leaves play probability alone (a Questionable player is assumed to play, the user's rule) and multiplies a
Questionable player's projection by a factor per position and cell. Factors are least squares through the origin of actual
on projected, relative to unlisted players, fit on the OTHER seasons; the held-out season is scored. Rules:

  current     what the board already does (WR x0.83 points and the WR stat cut; RB / TE / QB untouched): factor 1 on top
  position    one factor per position
  ongoing     per position, two cells: new this week / ongoing (on the previous game week's report too, or missed that game)
  ongoing+pr  per position, four cells: ongoing x final practice (full / not full)

Scored on the Questionable played rows: points MSE (and the change against 'current'), per season, and the main stats.
A rule passes the fast screen when it lowers points MSE in 4 of 5 seasons and pooled; the harness bar (START-ALL <= -0.003,
no START position worse than +0.002) comes after.

    python scripts/screen_questionable_cells.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PANEL = os.path.join('.sweeps', 'study_questionable_timing_panel.parquet')
OUT = os.path.join('.sweeps', 'screen_questionable_cells.txt')
POSITIONS = ('QB', 'RB', 'WR', 'TE')
STATS = {'RB': ('rushing_attempts', 'rushing_yards', 'receptions'), 'WR': ('targets', 'receptions', 'receiving_yards'),
         'TE': ('targets', 'receptions', 'receiving_yards'), 'QB': ('passing_yards',)}
SHRINK_ROWS = 40          # a cell's factor is pulled toward its position's factor with this many rows' weight


def _factor(a, x):
    a, x = np.asarray(a, float), np.asarray(x, float)
    ok = np.isfinite(a) & np.isfinite(x)
    d = (x[ok] ** 2).sum()
    return float((a[ok] * x[ok]).sum() / d) if d > 0 else np.nan


def cells(p, rule):
    ongoing = np.where(p['history'].eq('new this week'), 'new', 'ongoing')
    full = np.where(p['practice'].eq('full'), 'full', 'not full')
    if rule == 'position':
        return pd.Series('all', index=p.index)
    if rule == 'ongoing':
        return pd.Series(ongoing, index=p.index)
    if rule == 'ongoing+pr':
        return pd.Series(ongoing, index=p.index) + ' / ' + pd.Series(full, index=p.index)
    raise ValueError(rule)


def relative(p, col_actual, col_proj):
    """actual / the unlisted factor of the same season and position (so 1 = a healthy player's production)."""
    base = p[p['status'] == 'unlisted']
    f0 = {(s, pos): _factor(g[col_actual], g[col_proj]) for (s, pos), g in base.groupby(['season', 'Pos'])}
    return p[col_actual] / pd.Series([f0.get((s, pos), np.nan) for s, pos in zip(p['season'], p['Pos'])], index=p.index)


def fit_factors(train, rule, col_rel, col_proj):
    out = {}
    for pos, g in train.groupby('Pos'):
        fp = _factor(g[col_rel], g[col_proj])
        out[(pos, 'all')] = fp
        if rule != 'position':
            for c, h in g.groupby('_cell'):
                n = len(h)
                fc = _factor(h[col_rel], h[col_proj])
                out[(pos, c)] = (n * fc + SHRINK_ROWS * fp) / (n + SHRINK_ROWS) if np.isfinite(fc) else fp
    return out


def score(p, rule, col_actual='actual_pts', col_proj='model_pts'):
    """Per held-out season: rows, MSE under the current board and under the rule; the factors fit on all seasons."""
    q = p[p['status'] == 'Questionable'].copy()
    q['_rel'] = relative(p, col_actual, col_proj).loc[q.index]
    q = q[q['_rel'].notna() & q[col_proj].notna()]
    q['_cell'] = 'all' if rule == 'current' else cells(q, rule)
    rows = []
    for s in sorted(q['season'].unique()):
        test, train = q[q['season'] == s], q[q['season'] != s]
        if rule == 'current':
            f = pd.Series(1.0, index=test.index)
        else:
            fac = fit_factors(train, rule, '_rel', col_proj)
            f = pd.Series([fac.get((pos, c), fac.get((pos, 'all'), 1.0)) for pos, c in zip(test['Pos'], test['_cell'])],
                          index=test.index)
        base = ((test[col_actual] - test[col_proj]) ** 2).mean()
        new = ((test[col_actual] - test[col_proj] * f) ** 2).mean()
        rows.append({'season': s, 'rows': len(test), 'mse current': base, 'mse rule': new, 'change': new - base})
    full = {} if rule == 'current' else fit_factors(q, rule, '_rel', col_proj)
    return pd.DataFrame(rows), full


def main():
    p = pd.read_parquet(PANEL)
    p = p[p['Pos'].isin(POSITIONS)]
    lines = ['A1 screen: Questionable played rows, factors fit on the other seasons and scored on the held-out one (points). '
             f'Cell factors are shrunk toward the position factor ({SHRINK_ROWS} rows). Change < 0 = the rule is better.', '']
    for rule in ('position', 'ongoing', 'ongoing+pr'):
        t, fac = score(p, rule)
        pooled = (t['change'] * t['rows']).sum() / t['rows'].sum()
        lines += [f'RULE {rule}: pooled MSE change {pooled:+.3f} (current {(t["mse current"] * t["rows"]).sum() / t["rows"].sum():.2f}); '
                  f'better in {int((t["change"] < 0).sum())}/{len(t)} seasons',
                  t.to_string(index=False, float_format=lambda v: f'{v:.3f}'),
                  'factors (all seasons): ' + ', '.join(f'{k[0]} {k[1]} {v:.3f}' for k, v in sorted(fac.items())), '']
        for pos in POSITIONS:
            sub = p[(p['Pos'] == pos) | (p['status'] == 'unlisted')]
            sub = sub[sub['Pos'] == pos]
            t2, _ = score(sub, rule)
            pooled2 = (t2['change'] * t2['rows']).sum() / t2['rows'].sum()
            lines.append(f'   {pos}: rows {int(t2["rows"].sum())}, pooled change {pooled2:+.3f}, better in '
                         f'{int((t2["change"] < 0).sum())}/{len(t2)} seasons')
        lines.append('')
    for rule in ('position', 'ongoing'):
        lines.append(f"STATS under rule '{rule}' (stat factors fit the same way)")
        for pos, stats in STATS.items():
            sub = p[p['Pos'] == pos]
            for st in stats:
                if st not in sub.columns or f'actual_{st}' not in sub.columns:
                    continue
                s2 = sub.assign(**{st: pd.to_numeric(sub[st], errors='coerce'),
                                   f'actual_{st}': pd.to_numeric(sub[f'actual_{st}'], errors='coerce')})
                t3, fac3 = score(s2, rule, f'actual_{st}', st)
                pooled3 = (t3['change'] * t3['rows']).sum() / t3['rows'].sum()
                base3 = (t3['mse current'] * t3['rows']).sum() / t3['rows'].sum()
                lines.append(f'   {pos} {st}: pooled change {pooled3:+.3f} on MSE {base3:.2f} ({pooled3 / base3:+.1%}), '
                             f'better in {int((t3["change"] < 0).sum())}/{len(t3)}; factors '
                             + ', '.join(f'{k[1]} {v:.3f}' for k, v in sorted(fac3.items())))
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
