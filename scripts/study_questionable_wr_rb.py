"""
Questionable players who play, a closer look (2026-11-11, A1 follow-up; user: "refine the WR cut ... make sure .83 is the
correct value and not just some other data issue. For RB check ... if it is just a volume issue, or if the RB is just more
likely to get injured in that next game, or if their efficiency comes down").

A Questionable player is always assumed to play; only the production of players who did play is read.

WR. The shipped 0.83 (v2_questionable_workload) was fit on 2026-10-08 from .sweeps/calibration_lab_v7.csv (the pre-cut model,
2021-2025), with the injury report joined on name + week only. Here:
  1. reproduce it from that dump, then with the report joined on name + team + week (namesakes), and with the model's own
     replayed tag (the dump's 'inj' column, what the cut actually keys on);
  2. refit on today's model: the harness base boards 2022-2025 (which carry the cut) with the cut undone on the rows it touched
     (found from Model / Raw against untagged WRs at the same Raw level), plus the 2026 live ledger boards;
  3. probe: by season, by projection level, START pool vs all, without touchdowns, per stat, played defined by snaps (any,
     or 25%+ of expected), and on 2026 the FantasyPros tag (what the live board keys on) against the official report.

RB. On the same panel, Questionable RBs who played against untagged RBs who played, per season: snap share against expected,
carries, rushing yards, yards per carry, targets, receptions, receiving yards, touchdowns and points (each as actual / projected
relative to the untagged ratio); how often the game ended early (offence share under half of expected) and how often the next
game was missed (no snap, or Out / Doubtful / reserve at the next report); and the points factor with the early exits removed.

    python scripts/study_questionable_wr_rb.py [--out .sweeps/study_questionable_wr_rb.txt]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.historical_availability import historical_injury_profiles, load_injury_reports, load_weekly_rosters  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

DUMP = os.path.join('.sweeps', 'calibration_lab_v7.csv')
BOARD = os.path.join('.sweeps', 'harness_boards', 'capacity_screen_2022-2025_wk3-17.parquet')
HIST = (2022, 2023, 2024, 2025)
LIVE = 2026
CUT = 0.83
STAT_CUT = {'targets': 0.926, 'receptions': 0.888, 'receiving_yards': 0.862, 'receiving_tds': 0.564}
STATS = ('rushing_attempts', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds')
START_N = {'QB': 24, 'RB': 40, 'WR': 55, 'TE': 20}
RESERVE = {'RES', 'EXE', 'RET', 'CUT', 'TRD', 'DEV'}


def ls(a, x):
    a, x = np.asarray(a, float), np.asarray(x, float)
    ok = np.isfinite(a) & np.isfinite(x)
    d = (x[ok] ** 2).sum()
    return float((a[ok] * x[ok]).sum() / d) if d > 0 else np.nan


def ratio(a, x):
    a, x = np.asarray(a, float), np.asarray(x, float)
    ok = np.isfinite(a) & np.isfinite(x)
    return float(a[ok].sum() / x[ok].sum()) if x[ok].sum() > 0 else np.nan


# ---------------------------------------------------------------------------------------------------------------------
# 1. the original fit, reproduced
# ---------------------------------------------------------------------------------------------------------------------
def reproduce_dump():
    d = pd.read_csv(DUMP, low_memory=False)
    d = d[(d['pos'] == 'WR') & d['week'].between(3, 17)].copy()
    d['played'] = d['played'].astype(bool)
    d['start'] = d.groupby(['year', 'week'])['final'].rank(ascending=False, method='first') <= START_N['WR']
    d['_key'] = clean_name_exact(d['player'].astype(str))
    reps = []
    for y in sorted(d['year'].unique()):
        r = load_injury_reports(int(y))
        r = r.assign(year=int(y), _key=clean_name_exact(r['full_name'].astype(str)), week=r['week'].astype(int))
        reps.append(r[['year', 'week', '_key', 'team', 'report_status']].drop_duplicates(['year', 'week', '_key', 'team']))
    rep = pd.concat(reps)
    by_name = rep.drop_duplicates(['year', 'week', '_key'])[['year', 'week', '_key', 'report_status']]
    d = d.merge(by_name.rename(columns={'report_status': 'rep_name'}), on=['year', 'week', '_key'], how='left')
    d = d.merge(rep.rename(columns={'report_status': 'rep_team'}), on=['year', 'week', '_key', 'team'], how='left')
    d['tag_model'] = d['inj'].astype(str).str.startswith('Questionable')
    out = []
    healthy = d[d['played'] & d['start'] & d['rep_team'].isna() & ~d['tag_model']]
    h = ratio(healthy['actual'], healthy['final'])
    for label, mask in (('report, name + week (as shipped)', d['rep_name'].eq('Questionable')),
                        ('report, name + team + week', d['rep_team'].eq('Questionable')),
                        ("the model's own replayed tag", d['tag_model'])):
        g = d[mask & d['played'] & d['start']]
        out.append({'join': label, 'rows': len(g), 'actual/proj': ratio(g['actual'], g['final']),
                    'ls factor': ls(g['actual'], g['final']), 'untagged actual/proj': h,
                    'relative': ratio(g['actual'], g['final']) / h})
    diff = d[d['rep_name'].eq('Questionable') != d['rep_team'].eq('Questionable')]
    return pd.DataFrame(out), diff[['year', 'week', 'player', 'team', 'rep_name', 'rep_team']].head(20), len(diff)


# ---------------------------------------------------------------------------------------------------------------------
# 2. the panel on today's model
# ---------------------------------------------------------------------------------------------------------------------
def report_frame(season):
    r = load_injury_reports(season)
    r = r.assign(_key=clean_name_exact(r['full_name'].astype(str)), week=r['week'].astype(int))
    return r.drop_duplicates(['week', '_key', 'team'], keep='last')[['week', '_key', 'team', 'report_status', 'practice_status']]


def actuals(season):
    stats, _tc, name_col, _ = load_and_merge_data(season, 'Full PPR')
    stats = stats[stats['season'] == season] if 'season' in stats.columns else stats
    a = stats[['week', name_col, 'fantasy_points_ppr'] + [s for s in STATS if s in stats.columns]].copy()
    a['_key'] = clean_name_exact(a[name_col].astype(str))
    a = a.drop(columns=[name_col]).drop_duplicates(['week', '_key'])
    return a.rename(columns={'fantasy_points_ppr': 'actual_pts', **{s: f'actual_{s}' for s in STATS}})


def snaps(season):
    s = pd.read_parquet(os.path.join('data', 'cache', f'snap_counts_{season}.parquet'))
    s = s[s['game_type'].astype(str).eq('REG')].assign(_key=lambda x: clean_name_exact(x['player'].astype(str)))
    return s.groupby(['week', '_key', 'team'], as_index=False)[['offense_snaps', 'offense_pct']].max()


def next_game_missed(season, rows, rep, snp):
    """Per row: the player's team's next game week, and whether he missed it (no snap there, or Out / Doubtful at its
    report, or a reserve / cut status on the weekly roster)."""
    played_weeks = snp[snp['offense_snaps'] > 0][['week', 'team']].drop_duplicates()
    team_weeks = {t: sorted(g['week']) for t, g in played_weeks.groupby('team')}
    snapped = set(zip(snp.loc[snp['offense_snaps'] > 0, 'week'], snp.loc[snp['offense_snaps'] > 0, '_key']))
    out_next = set(zip(rep.loc[rep['report_status'].isin(['Out', 'Doubtful']), 'week'],
                       rep.loc[rep['report_status'].isin(['Out', 'Doubtful']), '_key']))
    ros = load_weekly_rosters(season)
    ros = ros.assign(_key=clean_name_exact(ros['full_name'].astype(str)))
    gone = set(zip(ros.loc[ros['status'].astype(str).str.upper().isin(RESERVE), 'week'].astype(int),
                   ros.loc[ros['status'].astype(str).str.upper().isin(RESERVE), '_key']))
    nxt, missed = [], []
    for w, k, t in zip(rows['week'], rows['_key'], rows['Team']):
        later = [x for x in team_weeks.get(t, []) if x > w]
        if not later:
            nxt.append(np.nan)
            missed.append(np.nan)
            continue
        n = later[0]
        nxt.append(n)
        missed.append(float((n, k) not in snapped or (n, k) in out_next or (n, k) in gone))
    return np.array(nxt), np.array(missed)


def detect_cut(p):
    """For WR rows: Model / (Raw x the untagged median Model / Raw at that Raw level); the cut rows read about 0.83."""
    w = p[(p['Pos'] == 'WR') & (p['Raw Model Proj Pts'] > 0.5)].copy()
    w['_r'] = w['Model Proj Pts'] / w['Raw Model Proj Pts']
    bins = pd.qcut(w['Raw Model Proj Pts'], 20, duplicates='drop')
    w['_bin'] = bins
    typical = w[w['status'] == 'unlisted'].groupby('_bin', observed=True)['_r'].median()
    rel = w['_r'] / w['_bin'].map(typical).astype(float)
    return rel.reindex(p.index)


def panel():
    board = pd.read_parquet(BOARD)
    board = board[(board['variant'] == 'base') & board['Pos'].isin(('RB', 'WR'))].copy()
    board['_key'] = clean_name_exact(board['Player'].astype(str))
    frames = []
    for season in HIST:
        b = board[board['year'] == season].copy()
        rep, snp = report_frame(season), snaps(season)
        b = b.merge(rep.rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
        b = b.merge(actuals(season), on=['week', '_key'], how='inner')
        b = b.merge(snp.rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
        b['next_week'], b['next_missed'] = next_game_missed(season, b, rep, snp)
        b['season'] = season
        b['tag_live'] = np.nan
        # the harness cut exactly as the model applies it: historical_injury_profiles keyed by the board's Player string
        sched = load_schedule(season)
        prof = {w: historical_injury_profiles(season, w, sched) for w in sorted(b['week'].unique())}
        tagged = np.array([str((prof[w].get(pl) or {}).get('status', '')).lower() == 'questionable'
                           for w, pl in zip(b['week'], b['Player'])])
        b['pts_cut'] = tagged
        b['stat_cut'] = tagged
        frames.append(b)
    # 2026 live boards
    from data import ledger_scorecard as sc
    stats, _tc, name_col, _ = load_and_merge_data(LIVE, 'Full PPR')
    f, _info = sc.load_scored_ledger(LIVE, stats, name_col, load_schedule(LIVE))
    f = f[f['pre_kickoff'] & f['week'].between(3, 17) & f['Pos'].isin(('RB', 'WR')) & f['actual_pts'].notna()].copy()
    rep, snp = report_frame(LIVE), snaps(LIVE)
    f = f.merge(rep.rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
    f = f.merge(snp.rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
    f['next_week'], f['next_missed'] = next_game_missed(LIVE, f, rep, snp)
    f['season'] = LIVE
    f['tag_live'] = f['Injury Status'].astype(str).str.startswith('Questionable')
    f['Expected Snap Share'] = np.nan
    # the live board cut its FantasyPros-Questionable WRs from the points cut's ship (2026-10-09 02:27 UTC) and their stat
    # line from the stat cut's (2026-11-03 01:36 UTC)
    built = pd.to_datetime(f['build_ts'], utc=True)
    f['pts_cut'] = f['tag_live'] & (built >= pd.Timestamp('2026-10-09 02:27', tz='UTC'))
    f['stat_cut'] = f['tag_live'] & (built >= pd.Timestamp('2026-11-03 01:36', tz='UTC'))
    keep = ['Player', 'Pos', 'Team', 'week', 'season', '_key', 'Availability', 'Raw Model Proj Pts', 'Model Proj Pts',
            'Expected Snap Share', 'report_status', 'practice_status', 'offense_snaps', 'offense_pct', 'next_week',
            'next_missed', 'tag_live', 'pts_cut', 'stat_cut', 'actual_pts'] + list(STATS) + [f'actual_{s}' for s in STATS]
    frames.append(f[[c for c in keep if c in f.columns]])
    p = pd.concat([x[[c for c in keep if c in x.columns]] for x in frames], ignore_index=True)
    p['status'] = np.where(p['report_status'].isna() & p['practice_status'].isna(), 'unlisted',
                           p['report_status'].fillna('listed, no status'))
    p['q'] = p['status'].eq('Questionable')
    # undo the shipped WR cut on the rows the board applied it to (apply_questionable_workload's own eligibility)
    p['cut_rel'] = detect_cut(p)                                         # kept as a check on the replay below
    live_pts = pd.to_numeric(p['Model Proj Pts'], errors='coerce')
    elig = (p['Pos'] == 'WR') & (pd.to_numeric(p['Availability'], errors='coerce').fillna(1.0) > 0.01) & (live_pts > 0.005)
    p['cut_applied'] = elig & p['pts_cut'].fillna(False).astype(bool)
    p['stat_cut_applied'] = elig & p['stat_cut'].fillna(False).astype(bool)
    p['pre_pts'] = np.where(p['cut_applied'], live_pts / CUT, live_pts)
    for s, fac in STAT_CUT.items():
        v = pd.to_numeric(p[s], errors='coerce')
        p[f'pre_{s}'] = np.where(p['stat_cut_applied'], v / fac, v)
    for s in STATS:
        if f'pre_{s}' not in p.columns:
            p[f'pre_{s}'] = pd.to_numeric(p[s], errors='coerce')
    p['proj_tds'] = p['pre_rushing_tds'].fillna(0) + p['pre_receiving_tds'].fillna(0)
    p['act_tds'] = p['actual_rushing_tds'].fillna(0) + p['actual_receiving_tds'].fillna(0)
    p['pre_pts_notd'] = p['pre_pts'] - 6 * p['proj_tds']
    p['act_pts_notd'] = p['actual_pts'] - 6 * p['act_tds']
    p['start'] = p.groupby(['season', 'week', 'Pos'])['pre_pts'].rank(ascending=False, method='first') <= \
        p['Pos'].map(START_N)                                             # ranked on the whole board, before 'played'
    p = p[(p['offense_snaps'].fillna(0) > 0) & (p['pre_pts'] > 0.5)].copy()   # played, and the board had him playing
    exp = pd.to_numeric(p['Expected Snap Share'], errors='coerce')
    p['early_exit'] = np.where(exp >= 0.3, p['offense_pct'] < 0.5 * exp, np.nan)
    return p


# ---------------------------------------------------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------------------------------------------------
def rel_factor(g, base, a, x, fn=ratio):
    """fn(actual, projected) on g, divided by the same on `base` season by season (weighted by g's projected sum)."""
    num = den = 0.0
    for s, h in g.groupby('season'):
        b = base[base['season'] == s]
        f0 = fn(b[a], b[x])
        if not np.isfinite(f0) or f0 == 0:
            continue
        w = pd.to_numeric(h[x], errors='coerce').sum()
        num += fn(h[a], h[x]) / f0 * w
        den += w
    return num / den if den else np.nan


def wr_tables(p):
    w = p[p['Pos'] == 'WR']
    base = w[w['status'] == 'unlisted']
    q = w[w['q']]
    lines = []
    hist = w[w['season'] < LIVE]
    lines.append("Check on the replayed cut (Model / the untagged Model at the same Raw level; the cut reads 0.83): "
                 f"replayed cut rows median {hist.loc[hist['cut_applied'], 'cut_rel'].median():.3f}, not cut "
                 f"{hist.loc[~hist['cut_applied'], 'cut_rel'].median():.3f}. Official-Questionable rows the model did not cut "
                 f"(name mismatch): {int((q['season'] < LIVE).sum() - (q['cut_applied'] & (q['season'] < LIVE)).sum())}; "
                 f"rows cut that are not official Questionable: {int((hist['cut_applied'] & ~hist['q']).sum())}. 2026: "
                 f"FantasyPros-Q {int(w['tag_live'].fillna(False).astype(bool).sum())}, points cut {int(w.loc[w['season'] == LIVE, 'cut_applied'].sum())}, "
                 f"stat cut {int(w.loc[w['season'] == LIVE, 'stat_cut_applied'].sum())}.")
    rows = []
    for label, mask, b in (
            ('all seasons, START', q['start'], base[base['start']]),
            ('all seasons, every row', pd.Series(True, index=q.index), base),
            ('START, without TDs', q['start'], base[base['start']]),
            ('START, LS through origin', q['start'], base[base['start']]),
            ('START, 25%+ of expected snaps', q['start'] & (q['offense_pct'] >= 0.25), base[base['start']]),
            ('START, no early exit', q['start'] & (q['early_exit'] != 1), base[base['start'] & (base['early_exit'] != 1)]),
            ('projection 10+', q['pre_pts'] >= 10, base[base['pre_pts'] >= 10]),
            ('projection 5-10', q['pre_pts'].between(5, 10), base[base['pre_pts'].between(5, 10)]),
            ('projection under 5', q['pre_pts'] < 5, base[base['pre_pts'] < 5])):
        g = q[mask]
        if 'without TDs' in label:
            f = rel_factor(g, b, 'act_pts_notd', 'pre_pts_notd')
        elif 'LS' in label:
            f = rel_factor(g, b, 'actual_pts', 'pre_pts', ls)
        else:
            f = rel_factor(g, b, 'actual_pts', 'pre_pts')
        rows.append({'cut fit on': label, 'rows': len(g), 'factor': f})
    mc = w[w['cut_applied'] & w['start']]
    rows.append({'cut fit on': "START, the rows the model cuts (its own tag)", 'rows': len(mc),
                 'factor': rel_factor(mc, base[base['start'] & ~base['cut_applied']], 'actual_pts', 'pre_pts')})
    for s in sorted(q['season'].unique()):
        g, b = q[q['start'] & (q['season'] == s)], base[base['start'] & (base['season'] == s)]
        rows.append({'cut fit on': f'season {s}, START', 'rows': len(g), 'factor': rel_factor(g, b, 'actual_pts', 'pre_pts')})
    # bootstrap over weeks
    g, b = q[q['start']], base[base['start']]
    rng = np.random.default_rng(0)
    weeks = g[['season', 'week']].drop_duplicates().to_numpy()
    boots = []
    for _ in range(1000):
        pick = weeks[rng.integers(0, len(weeks), len(weeks))]
        gb = pd.concat([g[(g['season'] == s) & (g['week'] == wk)] for s, wk in pick])
        boots.append(rel_factor(gb, b, 'actual_pts', 'pre_pts'))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows.append({'cut fit on': 'START, 95% interval (weeks resampled)', 'rows': len(g), 'factor': np.nan, 'low': lo, 'high': hi})
    stat_rows = []
    for s in ('targets', 'receptions', 'receiving_yards', 'receiving_tds'):
        stat_rows.append({'stat': s, 'shipped': STAT_CUT[s],
                          'refit START': rel_factor(q[q['start']], base[base['start']], f'actual_{s}', f'pre_{s}'),
                          'refit all': rel_factor(q, base, f'actual_{s}', f'pre_{s}')})
    stat_rows.append({'stat': 'snap share (actual / untagged, not a projection)', 'shipped': np.nan,
                      'refit START': q.loc[q['start'], 'offense_pct'].mean() / base.loc[base['start'], 'offense_pct'].mean(),
                      'refit all': np.nan})
    # 2026: FantasyPros tag vs official report
    live = w[w['season'] == LIVE]
    lb = live[live['status'] == 'unlisted']
    tags = []
    for label, mask in (('official Questionable', live['q']), ('FantasyPros Questionable', live['tag_live'].fillna(False).astype(bool)),
                        ('FP Q, not official Q', live['tag_live'].fillna(False).astype(bool) & ~live['q']),
                        ('official Q, not FP Q', live['q'] & ~live['tag_live'].fillna(False).astype(bool))):
        g = live[mask]
        tags.append({'2026 tag': label, 'rows': len(g), 'factor vs pre-cut': rel_factor(g, lb, 'actual_pts', 'pre_pts'),
                     'cut applied share': g['cut_applied'].mean() if len(g) else np.nan})
    return lines, pd.DataFrame(rows), pd.DataFrame(stat_rows), pd.DataFrame(tags)


def rb_tables(p):
    r = p[p['Pos'] == 'RB'].copy()
    r['ypc_proj'] = r['pre_rushing_yards'] / r['pre_rushing_attempts']
    base = r[r['status'] == 'unlisted']
    q = r[r['q']]
    out = []
    for label, a, x in (('points', 'actual_pts', 'pre_pts'), ('points without TDs', 'act_pts_notd', 'pre_pts_notd'),
                        ('carries', 'actual_rushing_attempts', 'pre_rushing_attempts'),
                        ('rushing yards', 'actual_rushing_yards', 'pre_rushing_yards'),
                        ('targets', 'actual_targets', 'pre_targets'), ('receptions', 'actual_receptions', 'pre_receptions'),
                        ('receiving yards', 'actual_receiving_yards', 'pre_receiving_yards'),
                        ('touchdowns', 'act_tds', 'proj_tds')):
        rec = {'measure': label, 'all': rel_factor(q, base, a, x)}
        for s in sorted(q['season'].unique()):
            rec[str(s)] = rel_factor(q[q['season'] == s], base[base['season'] == s], a, x)
        vals = [rec[str(s)] for s in sorted(q['season'].unique())]
        rec['seasons < 1'] = f"{sum(v < 1 for v in vals if np.isfinite(v))}/{sum(np.isfinite(v) for v in vals)}"
        out.append(rec)
    # yards per carry: actual yards / actual carries against the projection's yards / carries
    def ypc(g):
        return (g['actual_rushing_yards'].sum() / g['actual_rushing_attempts'].sum()) / \
            (g['pre_rushing_yards'].sum() / g['pre_rushing_attempts'].sum())
    rec = {'measure': 'yards per carry', 'all': ypc(q) / ypc(base)}
    for s in sorted(q['season'].unique()):
        rec[str(s)] = ypc(q[q['season'] == s]) / ypc(base[base['season'] == s])
    vals = [rec[str(s)] for s in sorted(q['season'].unique())]
    rec['seasons < 1'] = f"{sum(v < 1 for v in vals)}/{len(vals)}"
    out.append(rec)
    def ypt(g):
        return (g['actual_receiving_yards'].sum() / g['actual_targets'].sum()) / \
            (g['pre_receiving_yards'].sum() / g['pre_targets'].sum())
    out.append({'measure': 'receiving yards per target', 'all': ypt(q) / ypt(base),
                **{str(s): ypt(q[q['season'] == s]) / ypt(base[base['season'] == s]) for s in sorted(q['season'].unique())}})
    hist = r[r['season'] < LIVE]
    hq, hb = hist[hist['q']], hist[hist['status'] == 'unlisted']
    snap = {'measure': 'snap share vs expected (2022-2025)',
            'all': (hq['offense_pct'].sum() / hq['Expected Snap Share'].sum()) /
                   (hb['offense_pct'].sum() / hb['Expected Snap Share'].sum())}
    for s in HIST:
        a, b = hq[hq['season'] == s], hb[hb['season'] == s]
        snap[str(s)] = (a['offense_pct'].sum() / a['Expected Snap Share'].sum()) / (b['offense_pct'].sum() / b['Expected Snap Share'].sum())
    out.append(snap)
    decomp = pd.DataFrame(out)
    # injury in that game / the next
    inj = []
    for label, g in (('Questionable, played', q), ('untagged, played', base),
                     ('listed with no game status, played', r[r['status'] == 'listed, no status'])):
        h = g[g['season'] < LIVE]
        inj.append({'RB group': label, 'rows': len(g), 'early exit (2022-25)': h['early_exit'].mean(),
                    'missed next game': g['next_missed'].mean(),
                    'missed next | early exit': h.loc[h['early_exit'] == 1, 'next_missed'].mean(),
                    'missed next | no early exit': h.loc[h['early_exit'] == 0, 'next_missed'].mean()})
    injt = pd.DataFrame(inj)
    # the points factor with early exits removed, and among those who did not miss the next game
    alt = []
    for label, gq, gb in (('every Questionable game', q, base),
                          ('no early exit (2022-25)', hq[hq['early_exit'] == 0], hb[hb['early_exit'] == 0]),
                          ('early exit only (2022-25)', hq[hq['early_exit'] == 1], hb[hb['early_exit'] == 0]),
                          ('played the next game too', q[q['next_missed'] == 0], base[base['next_missed'] == 0]),
                          ('missed the next game', q[q['next_missed'] == 1], base[base['next_missed'] == 0])):
        alt.append({'RB subset': label, 'rows': len(gq), 'points factor': rel_factor(gq, gb, 'actual_pts', 'pre_pts'),
                    'carries factor': rel_factor(gq, gb, 'actual_rushing_attempts', 'pre_rushing_attempts')})
    return decomp, injt, pd.DataFrame(alt)


def heldout(p):
    """Held out by season (fit on the other seasons, score the held-out one), squared error on the Questionable rows who
    played; projections scaled by the held-out season's untagged actual / projected so only the Questionable factor differs."""
    lines = []

    def run(rows, base, a, x, shapes, label, ref_name):
        k = {s: g[a].sum() / g[x].sum() for s, g in base.groupby('season')}
        sse = {n: [] for n in shapes}
        for s in sorted(rows['season'].unique()):
            te, tr, btr = rows[rows['season'] == s], rows[rows['season'] != s], base[base['season'] != s]
            for n, fit in shapes.items():
                sse[n].append(((te[a] - fit(tr, btr, a, x)(te) * k[s]) ** 2).sum())
        ref = np.array(sse[ref_name])
        lines.append(f'{label} ({len(rows)} rows)')
        for n, v in sse.items():
            v = np.array(v)
            lines.append(f'   {n:34s} MSE {v.sum() / len(rows):8.3f}   vs {ref_name} {(v.sum() - ref.sum()) / len(rows):+.3f}   '
                         f'better in {(v < ref).sum()}/{len(v)} seasons')

    def flat(val):
        return lambda tr, b, a, x: (lambda te: te[x] * val)

    def refit(tr, b, a, x):
        f = rel_factor(tr, b, a, x)
        return lambda te: te[x] * f

    def ramp(t, cut_low=True):
        def fit(tr, b, a, x):
            hi = rel_factor(tr[tr[x] >= t], b[b[x] >= t], a, x)
            lo = rel_factor(tr[tr[x] < t], b[b[x] < t], a, x) if cut_low else 1.0
            return lambda te: te[x] * np.where(te[x] >= t, hi, lo)
        return fit

    w = p[p['Pos'] == 'WR']
    q, b = w[w['q']], w[w['status'] == 'unlisted']
    wr_shapes = {'shipped 0.83': flat(CUT), 'no cut': flat(1.0), 'flat, refit': refit,
                 'two factors, split at 7 pts': ramp(7), 'cut only 7+ pts': ramp(7, False)}
    run(q, b, 'actual_pts', 'pre_pts', wr_shapes, 'WR points, every Questionable WR who played', 'shipped 0.83')
    run(q[q['start']], b[b['start']], 'actual_pts', 'pre_pts', wr_shapes, 'WR points, START', 'shipped 0.83')
    for s, f in STAT_CUT.items():
        run(q, b, f'actual_{s}', f'pre_{s}', {'shipped': flat(f), 'no cut': flat(1.0), 'flat, refit': refit},
            f'WR {s}', 'shipped')
    r = p[p['Pos'] == 'RB'].copy()
    r['rush_pts'] = 0.1 * r['pre_rushing_yards'].fillna(0) + 6 * r['pre_rushing_tds'].fillna(0)
    q, b = r[r['q']], r[r['status'] == 'unlisted']

    def rush_only(tr, bb, a, x):
        fp = rel_factor(tr, bb, 'actual_pts', 'pre_pts')
        f = 1 - (1 - fp) / (tr['rush_pts'].sum() / tr['pre_pts'].sum())   # the same shortfall, all from the rushing line
        return lambda te: te['pre_pts'] - (1 - f) * te['rush_pts']

    run(q, b, 'actual_pts', 'pre_pts', {'no cut': flat(1.0), 'points x factor': refit, 'rushing line only x factor': rush_only},
        'RB points, every Questionable RB who played', 'no cut')
    for s in ('rushing_attempts', 'rushing_yards', 'receptions'):
        run(q, b, f'actual_{s}', f'pre_{s}', {'no cut': flat(1.0), 'flat, refit': refit}, f'RB {s}', 'no cut')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('.sweeps', 'study_questionable_wr_rb.txt'))
    a = ap.parse_args()
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    rep, diff, ndiff = reproduce_dump()
    p = panel()
    wl, wt, ws, wtag = wr_tables(p)
    rd, ri, ra = rb_tables(p)
    text = '\n'.join([
        "Questionable players who PLAYED (an offensive snap). factor = actual / projected (sums), relative to untagged players "
        "who played, same season; 1.00 = produces like a healthy player at that projection. WR projections are the PRE-cut "
        "ones (the shipped cut undone), so the WR factor is the cut itself.", '',
        'WR 1. THE SHIPPED FIT REPRODUCED (calibration_lab_v7, 2021-2025, START pool, pre-cut model)',
        rep.to_string(index=False, float_format=fmt),
        f'   rows tagged differently by the name-only join: {ndiff}', diff.to_string(index=False), '',
        "WR 2. REFIT ON TODAY'S MODEL (harness base boards 2022-2025 + live 2026)", *wl,
        wt.to_string(index=False, float_format=fmt), '',
        'WR 3. PER STAT (shipped v2_questionable_stat_cut against the refit)', ws.to_string(index=False, float_format=fmt), '',
        'WR 4. 2026: THE TAG THE LIVE BOARD READS (FantasyPros) AGAINST THE OFFICIAL REPORT', wtag.to_string(index=False, float_format=fmt), '',
        'RB 1. WHERE THE SHORTFALL IS (Questionable RBs who played, against untagged RBs who played)',
        rd.to_string(index=False, float_format=fmt), '',
        'RB 2. HURT IN THAT GAME OR THE NEXT (early exit = offence share under half of expected; next game missed = no snap, or '
        'Out / Doubtful / reserve)', ri.to_string(index=False, float_format=fmt), '',
        'RB 3. THE POINTS FACTOR BY WHAT HAPPENED', ra.to_string(index=False, float_format=fmt), '',
        'HELD OUT BY SEASON (2022-2026; fit on the other seasons)', *heldout(p)])
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    p.to_parquet(a.out.replace('.txt', '_panel.parquet'), index=False)


if __name__ == '__main__':
    main()
