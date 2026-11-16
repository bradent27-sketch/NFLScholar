"""
Receiver role drift (2026-11-16; user: "some sort of detection, or recency weight for receiver role change, both overall play
(snaps) but more importantly change in alignment (WR and TE) ... injury or scheme change in season can change whether a player is
playing more from slot or out wide or in line"). Offline, read only.

Question: when a WR / TE's role moves during the season - how many of his team's dropbacks he runs a route on (route
participation), or where he lines up (slot / wide / in line) - does the board lag it? If the miss (actual - board) tracks the
recent move, there is a lag to fix; if not, the model already follows the role through its per-game recency weights.

Data, 2024-2026 (PFF weekly receiving exports exist for these seasons only):
  board     2024-2025: the `base` arm of the newest harness board (the default model, weeks 3-17); 2026: the live pre-kickoff
            ledger boards weeks 3+ (data.ledger_scorecard)
  actuals   box scores (targets, receptions, receiving yards, receiving TDs, PPR points)
  role      pff_imports/<season>/weekly/<week>/receiving_summary.csv: routes, pass_plays, slot / wide / inline snaps per game;
            a team's dropbacks in a game = the most pass plays any of its receivers was on the field for
For each board row (played: 1+ snap that week), from his earlier games this season for the same team only:
  rp_season / rp_recent   routes / team dropbacks, season to date / last two games (ew_h: exponential, half-life h games)
  slot_*, wide_*, inline_*  the alignment mix (slot+wide+inline snaps), the same windows
  d_*                      recent minus season
The board's own per-game recency (0.85 per game) and prior-season blend are what is being tested against.

Reads: the miss by quintile of each drift (per season; a lag shows as a slope), and held-out corrections (fit on the other
seasons, scored on each): targets x (rp_ew / rp_season)^a; and detection of a move in the latest game (rises / drops,
WR / TE, thresholds 0.10-0.25), the receiving line x (rp_last / rp_before)^a - the TE rise at 0.15 became v2_te_route_rise.

    python scripts/study_role_alignment_drift.py [--out .sweeps/study_role_alignment_drift.txt]
"""
import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.pff_alignment import _canonical_team_key  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

BOARD = os.path.join('.sweeps', 'harness_boards', 'questionable_rb_2022-2025_wk3-17.parquet')
SEASONS = (2024, 2025, 2026)
LIVE = 2026
POS = ('WR', 'TE')
STATS = ('targets', 'receptions', 'receiving_yards', 'receiving_tds')
HALF_LIVES = (1.0, 2.0, 3.0, 4.0, 6.0)


def pff_weekly(season):
    rows = []
    for path in glob.glob(os.path.join('pff_imports', str(season), 'weekly', '*', 'receiving_summary.csv')):
        wk = int(os.path.basename(os.path.dirname(path)))
        d = pd.read_csv(path, low_memory=False)
        d = d.assign(week=wk)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d['Team'] = d['team_name'].map(_canonical_team_key)
    d['_key'] = clean_name_exact(d['player'].astype(str))
    for c in ('routes', 'pass_plays', 'slot_snaps', 'wide_snaps', 'inline_snaps', 'targets'):
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    db = d.groupby(['week', 'Team'])['pass_plays'].max().rename('team_db')
    d = d.join(db, on=['week', 'Team'])
    d['season'] = season
    return d[['season', 'week', 'Team', '_key', 'position', 'routes', 'pass_plays', 'slot_snaps', 'wide_snaps', 'inline_snaps',
              'targets', 'team_db']].rename(columns={'targets': 'pff_targets'})


def role_features(pff):
    """One row per (season, week, Team, _key) for every week a player could be projected: features from earlier games only."""
    out = []
    for (s, t, k), g in pff.groupby(['season', 'Team', '_key']):
        g = g.sort_values('week')
        weeks = g['week'].to_numpy()
        for target in range(2, 19):
            past = g[g['week'] < target]
            if past.empty:
                continue
            rec = {'season': s, 'Team': t, '_key': k, 'week': target, 'games': len(past)}
            align_tot = past['slot_snaps'] + past['wide_snaps'] + past['inline_snaps']
            n = np.arange(len(past))[::-1]               # games ago: 0 = most recent

            def window(weights):
                w = np.asarray(weights, float)
                db = (past['team_db'] * w).sum()
                at = (align_tot * w).sum()
                return {'rp': (past['routes'] * w).sum() / db if db > 0 else np.nan,
                        'tprr': (past['pff_targets'] * w).sum() / max((past['routes'] * w).sum(), 1e-9),
                        'slot': (past['slot_snaps'] * w).sum() / at if at > 0 else np.nan,
                        'wide': (past['wide_snaps'] * w).sum() / at if at > 0 else np.nan,
                        'inline': (past['inline_snaps'] * w).sum() / at if at > 0 else np.nan}
            for name, w in (('season', np.ones(len(past))), ('recent', (n < 2).astype(float))):
                for f, v in window(w).items():
                    rec[f'{f}_{name}'] = v
            for h in HALF_LIVES:
                for f, v in window(0.5 ** (n / h)).items():
                    rec[f'{f}_ew{h:g}'] = v
            rec['last_week_played'] = int(weeks[weeks < target].max())
            out.append(rec)
    return pd.DataFrame(out)


def actuals(season):
    stats, _tc, name_col, _ = load_and_merge_data(season, 'Full PPR')
    stats = stats[stats['season'] == season] if 'season' in stats.columns else stats
    keep = ['week', name_col, 'fantasy_points_ppr', 'weekly_snap_pct'] + [s for s in STATS if s in stats.columns]
    a = stats[keep].copy()
    a['_key'] = clean_name_exact(a[name_col].astype(str))
    a = a.drop(columns=[name_col]).drop_duplicates(['week', '_key'])
    return a.rename(columns={'fantasy_points_ppr': 'actual_pts', 'weekly_snap_pct': 'actual_snap',
                             **{s: f'actual_{s}' for s in STATS}})


def boards():
    b = pd.read_parquet(BOARD)
    b = b[(b['variant'] == 'base') & b['Pos'].isin(POS) & b['year'].isin(SEASONS)].copy()
    b = b.rename(columns={'year': 'season'})
    b['_key'] = clean_name_exact(b['Player'].astype(str))
    frames = []
    for s in sorted(b['season'].unique()):
        frames.append(b[b['season'] == s].merge(actuals(s), on=['week', '_key'], how='inner'))
    from data import ledger_scorecard as sc
    from data.loaders import load_schedule
    stats, _tc, name_col, _ = load_and_merge_data(LIVE, 'Full PPR')
    f, _ = sc.load_scored_ledger(LIVE, stats, name_col, load_schedule(LIVE))
    f = f[f['pre_kickoff'] & (f['week'] >= 3) & f['Pos'].isin(POS) & f['actual_pts'].notna()].copy()
    f['season'] = LIVE
    f = f.merge(actuals(LIVE)[['week', '_key', 'actual_snap']], on=['week', '_key'], how='left')
    frames.append(f)
    keep = ['Player', 'Pos', 'Team', 'season', 'week', '_key', 'Model Proj Pts', 'actual_pts', 'actual_snap'] + \
        list(STATS) + [f'actual_{s}' for s in STATS]
    p = pd.concat([x[[c for c in keep if c in x.columns]] for x in frames], ignore_index=True)
    for c in keep[6:]:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    return p[(p['actual_snap'] > 0) & (p['Model Proj Pts'] > 0.5)]


def quintile_table(p, drift, res, label):
    rows = []
    for s, g in p.dropna(subset=[drift]).groupby('season'):
        q = pd.qcut(g[drift], 5, labels=False, duplicates='drop')
        m = g.groupby(q)[res].mean()
        rows.append({'season': s, **{f'q{int(i) + 1}': v for i, v in m.items()},
                     'slope q5-q1': m.iloc[-1] - m.iloc[0], 'rows': len(g)})
    t = pd.DataFrame(rows)
    return f'{label}: miss ({res}) by quintile of {drift} (q1 = moved down most, q5 = moved up most)\n' + \
        t.to_string(index=False, float_format=lambda v: f'{v:+.3f}')


def heldout_scale(p, stat, ratio_col, alphas=np.linspace(-0.5, 1.5, 21)):
    """board stat x ratio^a, a fit on the other seasons (least squares); the held-out MSE change per season."""
    d = p.dropna(subset=[ratio_col]).copy()
    d = d[np.isfinite(d[ratio_col]) & (d[ratio_col] > 0)]
    d['_r'] = d[ratio_col].clip(0.25, 4.0)
    out = []
    for s in sorted(d['season'].unique()):
        tr, te = d[d['season'] != s], d[d['season'] == s]
        mse = [((tr[f'actual_{stat}'] - tr[stat] * tr['_r'] ** a) ** 2).mean() for a in alphas]
        a = alphas[int(np.argmin(mse))]
        base = ((te[f'actual_{stat}'] - te[stat]) ** 2).mean()
        new = ((te[f'actual_{stat}'] - te[stat] * te['_r'] ** a) ** 2).mean()
        out.append({'season': s, 'alpha (fit elsewhere)': a, 'rows': len(te), 'mse board': base,
                    'change': new - base, 'change %': (new - base) / base})
    return pd.DataFrame(out)


def latest_move_features(pff, min_before=2):
    """The v2_te_route_rise inputs per board week: latest game's route participation vs the pooled earlier games."""
    rows = []
    for (s, t, k), g in pff.groupby(['season', 'Team', '_key']):
        g = g.sort_values('week')
        for target in range(3, 19):
            past = g[g['week'] < target]
            if len(past) < min_before + 1:
                continue
            last, before = past.iloc[-1], past.iloc[:-1]
            if last['team_db'] <= 0 or before['team_db'].sum() <= 0:
                continue
            rows.append({'season': s, 'Team': t, '_key': k, 'week': target,
                         'rp_last': last['routes'] / last['team_db'],
                         'rp_before': before['routes'].sum() / before['team_db'].sum()})
    return pd.DataFrame(rows)


def rule_score(d, mask, a_grid=np.round(np.arange(0, 1.51, 0.05), 2), cap=1.5):
    """Receiving line x min((rp_last / rp_before)^a, cap) on the masked rows, a fit by PPR points MSE on the other seasons;
    the held-out points SSE change per season on the moved rows, the START rows and every row."""
    d = d.copy()
    d['ratio'] = (d['rp_last'] / d['rp_before']).clip(0.2, 5.0)

    def pts(g, a):
        f = np.minimum(g['ratio'] ** a, cap)
        return g['Model Proj Pts'] + (g['receptions'] + 0.1 * g['receiving_yards'] + 6 * g['receiving_tds']) * (f - 1)
    out = []
    for s in sorted(d['season'].unique()):
        tr, te, m_te = d[(d['season'] != s) & mask], d[d['season'] == s], mask[d['season'] == s].to_numpy()
        a = a_grid[int(np.argmin([((tr['actual_pts'] - pts(tr, a)) ** 2).mean() for a in a_grid]))]
        new = np.where(m_te, pts(te, a), te['Model Proj Pts'])
        for pool, m in (('rows', m_te), ('START', te['start'].to_numpy()), ('all', np.ones(len(te), bool))):
            base = ((te['actual_pts'] - te['Model Proj Pts'])[m] ** 2).sum()
            out.append({'season': s, 'pool': pool, 'a': a, 'n': int(m.sum()),
                        'dsse': ((te['actual_pts'] - new)[m] ** 2).sum() - base, 'sse0': base})
    return pd.DataFrame(out)


def rule_lines(p, pff):
    """Detection reads: rises / drops of the latest game's route participation, WR and TE, thresholds 0.10-0.25."""
    start_n = {'WR': 55, 'TE': 20}
    p = p.copy()
    p['start'] = p.groupby(['season', 'week', 'Pos'])['Model Proj Pts'].rank(ascending=False, method='first') <= \
        p['Pos'].map(start_n)
    f = latest_move_features(pff)
    lines = ['Detection (latest game vs his pooled earlier games, 2+; held out by season; dMSE = points MSE change; the TE',
             'rise at 0.15 is v2_te_route_rise):']
    for pos in POS:
        d = p[p['Pos'] == pos].drop(columns=['rp_last', 'rp_before'], errors='ignore').merge(
            f, on=['season', 'Team', '_key', 'week'], how='inner')
        dev = d['rp_last'] - d['rp_before']
        for direction in ('up', 'down'):
            for thr in (0.10, 0.15, 0.20, 0.25):
                mask = (dev >= thr) if direction == 'up' else (dev <= -thr)
                t = rule_score(d, mask)
                parts = []
                for pool in ('rows', 'START', 'all'):
                    h = t[t['pool'] == pool]
                    parts.append(f"{pool} n {h['n'].sum():5d} {h['dsse'].sum() / h['sse0'].sum():+.2%} "
                                 f"({'/'.join(f'{v:+.1%}' for v in h['dsse'] / h['sse0'])})")
                a = '/'.join(f'{v:.2f}' for v in t.loc[t['pool'] == 'all', 'a'])
                lines.append(f'   {pos} {direction:4s} {thr:.2f}  a {a} | ' + ' | '.join(parts))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('.sweeps', 'study_role_alignment_drift.txt'))
    args = ap.parse_args()
    pff = pd.concat([pff_weekly(s) for s in SEASONS], ignore_index=True)
    feats = role_features(pff)
    p = boards().merge(feats, on=['season', 'Team', '_key', 'week'], how='left')
    p = p[p['games'] >= 2].copy()
    p['res_targets'] = p['actual_targets'] - p['targets']
    p['res_receptions'] = p['actual_receptions'] - p['receptions']
    p['res_yards'] = p['actual_receiving_yards'] - p['receiving_yards']
    p['res_pts'] = p['actual_pts'] - p['Model Proj Pts']
    p['d_rp'] = p['rp_recent'] - p['rp_season']
    p['d_slot'] = p['slot_recent'] - p['slot_season']
    p['d_inline'] = p['inline_recent'] - p['inline_season']
    p['d_wide'] = p['wide_recent'] - p['wide_season']
    for h in HALF_LIVES:
        p[f'rp_ratio_ew{h:g}'] = p[f'rp_ew{h:g}'] / p['rp_season']
    p['rp_ratio_recent'] = p['rp_recent'] / p['rp_season']
    lines = [f'Rows: WR/TE who played, 2+ earlier PFF games this season for the same team: '
             f'{p.groupby(["season", "Pos"]).size().to_dict()}', '',
             'Role moves are common? share of rows with |d_rp| > 0.10: '
             f'{(p["d_rp"].abs() > 0.10).mean():.1%}; |d_slot| > 0.15: {(p["d_slot"].abs() > 0.15).mean():.1%}; '
             f'TE |d_inline| > 0.15: {(p.loc[p.Pos == "TE", "d_inline"].abs() > 0.15).mean():.1%}', '']
    for pos in POS:
        q = p[p['Pos'] == pos]
        lines += [f'===== {pos}',
                  quintile_table(q, 'd_rp', 'res_targets', 'route participation'),
                  quintile_table(q, 'd_rp', 'res_pts', 'route participation'),
                  quintile_table(q, 'd_slot', 'res_targets', 'slot share'),
                  quintile_table(q, 'd_slot', 'res_receptions', 'slot share'),
                  quintile_table(q, 'd_slot', 'res_yards', 'slot share'),
                  quintile_table(q, 'd_slot', 'res_pts', 'slot share')]
        if pos == 'TE':
            lines += [quintile_table(q, 'd_inline', 'res_targets', 'in-line share'),
                      quintile_table(q, 'd_inline', 'res_pts', 'in-line share')]
        lines.append('')
        lines.append(f'{pos} held out: targets x (route participation ratio)^a')
        for col in ['rp_ratio_recent'] + [f'rp_ratio_ew{h:g}' for h in HALF_LIVES]:
            t = heldout_scale(q, 'targets', col)
            lines.append(f'   {col:18s} ' + '; '.join(f"{int(r.season)}: a {r['alpha (fit elsewhere)']:+.1f} "
                                                     f"{r['change %']:+.2%}" for _, r in t.iterrows())
                         + f"  | pooled {(t['change'] * t['rows']).sum() / (t['mse board'] * t['rows']).sum():+.2%}")
        lines.append('')
    lines += rule_lines(p, pff)
    text = '\n'.join(lines)
    print(text)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    p.to_parquet(args.out.replace('.txt', '_panel.parquet'), index=False)


if __name__ == '__main__':
    main()
