"""
A1 follow-up (2026-11-11): when does a Questionable tag mean less production? The user's frame: a Questionable player is
assumed to play (no play-probability cut); the signal is how much less he produces when he does. So only rows where the player
played count, and the read is his production against the projection, relative to unlisted players the same season.

Panel, played rows (an offensive snap, or a box score in 2026):
  2022-2025  the harness base boards (.sweeps/harness_boards/capacity_screen_2022-2025_wk3-17.parquet, variant 'base': the
             default of 2026-11-03, which already carries the WR Questionable cut v2_questionable_workload /
             v2_questionable_stat_cut), nflverse injury report (the week's last filing), box scores, snap counts;
  2026       the live pre-kickoff ledger boards (data/ledger_scorecard.py), weeks 3-9, the same injury feed, and the
             FantasyPros day-by-day practice report (external_data/fantasypros_injury_archive, practice_1-3).

Splits of the Questionable rows: final practice status (full / limited / DNP); listing history (new this week; listed the
previous game week and played; missed the previous game - Out, Doubtful or no snap); weeks in a row on the report; injury
type; and, 2026 only, the Wed-Thu-Fri pattern.

Measures (per position and split): rows; the production factor = least squares through the origin of actual on projected
points, divided by the same for unlisted players (same season and position), so 1.00 = produces like a healthy player at that
projection; the mean points bias relative to unlisted players; the same factor on the position's main stats; per-season factors
and how many seasons sit below 1.

    python scripts/study_questionable_timing.py [--out .sweeps/study_questionable_timing.txt]
"""
import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.historical_availability import load_injury_reports  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402

BOARD = os.path.join('.sweeps', 'harness_boards', 'capacity_screen_2022-2025_wk3-17.parquet')
HIST_SEASONS = (2022, 2023, 2024, 2025)
LIVE_SEASON = 2026
POSITIONS = ('QB', 'RB', 'WR', 'TE')
MAIN_STATS = {'QB': ('passing_yards', 'rushing_yards'), 'RB': ('rushing_attempts', 'rushing_yards', 'receptions'),
              'WR': ('targets', 'receptions', 'receiving_yards'), 'TE': ('targets', 'receptions', 'receiving_yards')}
ALL_STATS = sorted({s for v in MAIN_STATS.values() for s in v})
PRACTICE = {'Full Participation in Practice': 'full', 'Limited Participation in Practice': 'limited',
            'Did Not Participate In Practice': 'DNP'}
FP_PRACTICE = {'DNP': 'D', 'Limit': 'L', 'Limited': 'L', 'Full': 'F'}
INJURY_GROUPS = {
    'soft tissue': ('hamstring', 'calf', 'groin', 'quad', 'thigh', 'pectoral', 'oblique', 'abdomen', 'adductor'),
    'ankle / foot': ('ankle', 'foot', 'toe', 'heel', 'achilles'),
    'knee': ('knee', 'acl', 'mcl', 'meniscus'),
    'upper body': ('shoulder', 'arm', 'hand', 'wrist', 'finger', 'thumb', 'elbow', 'forearm', 'bicep', 'tricep', 'collarbone'),
    'back / core': ('back', 'hip', 'rib', 'chest', 'core', 'neck'),
    'head': ('concussion', 'head'),
    'illness / rest': ('illness', 'rest', 'personal', 'not injury', 'non-injury', 'flu'),
}
MIN_ROWS = 25


def injury_group(text):
    t = str(text or '').lower()
    for group, words in INJURY_GROUPS.items():
        if any(w in t for w in words):
            return group
    return 'other' if t and t != 'nan' else 'unknown'


def injury_features(season):
    """Per (week, _key, team): final game status, final practice status, injury group, listing history and streak."""
    inj = load_injury_reports(season)
    if inj is None or inj.empty:
        return pd.DataFrame()
    inj = inj.copy()
    inj['_key'] = clean_name_exact(inj['full_name'].astype(str))
    inj['week'] = inj['week'].astype(int)
    inj = inj.drop_duplicates(['week', '_key', 'team'], keep='last')
    inj['practice'] = inj['practice_status'].map(PRACTICE).fillna('none')
    inj['injury'] = inj['report_primary_injury'].fillna(inj['practice_primary_injury']).map(injury_group)
    inj = inj.sort_values(['_key', 'team', 'week'])
    prev_week = inj.groupby(['_key', 'team'])['week'].shift()
    prev_status = inj.groupby(['_key', 'team'])['report_status'].shift()
    inj['prev_listed_week'] = prev_week
    inj['prev_status'] = prev_status
    # weeks in a row on the report (a bye week does not break the run)
    run, last, out = {}, {}, []
    for k, t, w in zip(inj['_key'], inj['team'], inj['week']):
        key = (k, t)
        run[key] = run.get(key, 0) + 1 if key in last and w - last[key] <= 2 else 1
        last[key] = w
        out.append(run[key])
    inj['streak'] = out
    return inj[['week', '_key', 'team', 'report_status', 'practice', 'injury', 'prev_listed_week', 'prev_status', 'streak']]


def fp_practice_2026():
    """{(week, _key): 'DLF'-style pattern of the week's practice days} from the archived FantasyPros pulls (latest per week)."""
    out = {}
    for path in sorted(glob.glob(os.path.join('external_data', 'fantasypros_injury_archive', f'{LIVE_SEASON}_wk*.json'))):
        try:
            with open(path, encoding='utf-8') as fh:
                snap = json.load(fh)
        except (OSError, ValueError):
            continue
        for i in snap.get('injuries') or []:
            days = [FP_PRACTICE.get(str(i.get(f'practice_{d}') or ''), '') for d in (1, 2, 3)]
            if any(days):
                out[(int(snap['week']), clean_name_exact(pd.Series([str(i.get('name') or '')]))[0])] = ''.join(
                    d or '-' for d in days)
    return out


def history_rows():
    board = pd.read_parquet(BOARD)
    board = board[(board['variant'] == 'base') & board['Pos'].isin(POSITIONS)].copy()
    board['_key'] = clean_name_exact(board['Player'].astype(str))
    frames = []
    for season in HIST_SEASONS:
        b = board[board['year'] == season].copy()
        stats, _tc, name_col, _ = load_and_merge_data(season, 'Full PPR')
        stats = stats[stats['season'] == season] if 'season' in stats.columns else stats
        act = stats[['week', name_col, 'fantasy_points_ppr'] + [s for s in ALL_STATS if s in stats.columns]].copy()
        act['_key'] = clean_name_exact(act[name_col].astype(str))
        act = act.drop(columns=[name_col]).drop_duplicates(['week', '_key'])
        act = act.rename(columns={'fantasy_points_ppr': 'actual_pts', **{s: f'actual_{s}' for s in ALL_STATS}})
        b = b.merge(act, on=['week', '_key'], how='inner')
        snaps = pd.read_parquet(os.path.join('data', 'cache', f'snap_counts_{season}.parquet'))
        snaps = snaps[snaps['game_type'].astype(str).eq('REG')].assign(_key=lambda d: clean_name_exact(d['player'].astype(str)))
        snaps = snaps.groupby(['week', '_key'], as_index=False)[['offense_snaps', 'offense_pct']].max()
        b = b.merge(snaps, on=['week', '_key'], how='left')
        b = b[b['offense_snaps'].fillna(0) > 0]
        b['model_pts'] = pd.to_numeric(b['Model Proj Pts'], errors='coerce')
        b['snap_res'] = b['offense_pct'] - pd.to_numeric(b['Expected Snap Share'], errors='coerce')
        b['season'] = season
        frames.append(b.merge(injury_features(season).rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'],
                              how='left'))
    return pd.concat(frames, ignore_index=True)


def live_rows():
    from data import ledger_scorecard as sc
    from data.loaders import load_schedule
    stats, _tc, name_col, _ = load_and_merge_data(LIVE_SEASON, 'Full PPR')
    f, _info = sc.load_scored_ledger(LIVE_SEASON, stats, name_col, load_schedule(LIVE_SEASON))
    f = f[f['pre_kickoff'] & f['week'].between(3, 17) & f['Pos'].isin(POSITIONS)].copy()
    f = f[f['actual_pts'].notna()]
    f['season'] = LIVE_SEASON
    f['snap_res'] = np.nan
    f = f.merge(injury_features(LIVE_SEASON).rename(columns={'team': 'Team'}), on=['week', '_key', 'Team'], how='left')
    pattern = fp_practice_2026()
    f['fp_pattern'] = [pattern.get((int(w), k), '') for w, k in zip(f['week'], f['_key'])]
    return f


def label_rows(p):
    p = p.copy()
    p['status'] = np.where(p['report_status'].isna() & p['practice'].isna(), 'unlisted',
                           p['report_status'].fillna('listed, no status'))
    gap = p['week'] - p['prev_listed_week']
    prev_missed = p['prev_status'].isin(['Out', 'Doubtful'])
    p['history'] = np.where(gap.isna() | (gap > 2), 'new this week',
                            np.where(prev_missed, 'missed last game', 'listed last game too'))
    p['streak_band'] = np.where(p['streak'].fillna(0) >= 3, '3+ weeks', p['streak'].fillna(0).astype(int).astype(str) + ' week')
    return p


def _factor(actual, proj):
    a, x = np.asarray(actual, float), np.asarray(proj, float)
    ok = np.isfinite(a) & np.isfinite(x)
    d = float((x[ok] ** 2).sum())
    return float((a[ok] * x[ok]).sum() / d) if d > 0 else np.nan


def split_table(p, by, pos_list=POSITIONS):
    """Per position and value of `by` (Questionable rows): rows, production factor relative to unlisted (points and main stats),
    points bias relative to unlisted, snap residual relative, per-season points factors and seasons below 1."""
    out = []
    base = p[p['status'] == 'unlisted']
    for pos in pos_list:
        bpos = base[base['Pos'] == pos]
        f0 = {s: _factor(g['actual_pts'], g['model_pts']) for s, g in bpos.groupby('season')}
        b0 = {s: (g['actual_pts'] - g['model_pts']).mean() for s, g in bpos.groupby('season')}
        sn0 = {s: g['snap_res'].mean() for s, g in bpos.groupby('season')}
        st0 = {(s, st): _factor(g[f'actual_{st}'], g[st]) for s, g in bpos.groupby('season') for st in MAIN_STATS[pos]
               if st in g.columns and f'actual_{st}' in g.columns}
        q = p[(p['Pos'] == pos) & (p['status'] == 'Questionable')]
        for val, g in q.groupby(by):
            if len(g) < MIN_ROWS:
                continue
            rel = g['actual_pts'] / g['season'].map(f0)
            rec = {'Pos': pos, by: val, 'rows': len(g), 'proj': g['model_pts'].mean(),
                   'factor': _factor(rel, g['model_pts']),
                   'pts bias': float((g['actual_pts'] - g['model_pts'] - g['season'].map(b0)).mean()),
                   'snap': float((g['snap_res'] - g['season'].map(sn0)).mean()) if g['snap_res'].notna().any() else np.nan}
            for st in MAIN_STATS[pos]:
                if st in g.columns and f'actual_{st}' in g.columns:
                    den = pd.Series([st0.get((s, st), np.nan) for s in g['season']], index=g.index)
                    rec[f'{st} f'] = _factor(pd.to_numeric(g[f'actual_{st}'], errors='coerce') / den,
                                             pd.to_numeric(g[st], errors='coerce'))
            per = {s: _factor(h['actual_pts'] / f0.get(s, np.nan), h['model_pts']) for s, h in g.groupby('season')
                   if len(h) >= 5}
            rec['by season'] = ' '.join(f"{str(s)[2:]}:{v:.2f}" for s, v in sorted(per.items()))
            rec['seasons < 1'] = f"{sum(v < 1 for v in per.values())}/{len(per)}"
            out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('.sweeps', 'study_questionable_timing.txt'))
    a = ap.parse_args()
    p = label_rows(pd.concat([history_rows(), live_rows()], ignore_index=True))
    p.to_parquet(a.out.replace('.txt', '_panel.parquet'), index=False)
    q = p[p['status'] == 'Questionable']
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    p['all'] = 'all Questionable'
    sections = [
        "Questionable players who PLAYED, 2022-2025 harness base boards + 2026 live ledger boards (weeks 3-17 / 3-9). factor = "
        "actual / projected points (least squares through the origin) relative to unlisted players at the same position and "
        "season: 1.00 = produces like a healthy player; WR rows already carry the shipped 0.83 cut, so a WR factor of 1.00 "
        "means the cut is right. pts bias = actual minus projection, relative to unlisted. snap = offense share minus expected, "
        f"relative (2022-2025 only). Groups need {MIN_ROWS}+ rows.",
        f"Questionable played rows: {len(q):,} ({q.groupby('Pos').size().to_dict()})", '']
    for title, by in (('0. ALL QUESTIONABLE', 'all'), ('1. FINAL PRACTICE STATUS (the last report of the week)', 'practice'),
                      ('2. LISTING HISTORY', 'history'), ('3. WEEKS IN A ROW ON THE REPORT', 'streak_band'),
                      ('4. INJURY TYPE', 'injury')):
        sections += [title, split_table(p, by).to_string(index=False, float_format=fmt), '']
    p['practice x history'] = p['practice'].astype(str) + ' / ' + p['history'].astype(str)
    sections += ['5. FINAL PRACTICE x HISTORY', split_table(p, 'practice x history').to_string(index=False, float_format=fmt), '']
    live = p[(p['season'] == LIVE_SEASON)]
    lq = live[live['status'] == 'Questionable']
    pat = lq['fp_pattern'].replace('', 'no report')
    sections += [f"6. 2026 WED-THU-FRI PATTERN (FantasyPros; D = did not practise, L = limited, F = full); {len(lq)} "
                 "Questionable played rows, all positions pooled (too few to split by position)",
                 pat.value_counts().head(15).to_string(), '']
    lq = lq.assign(early_dnp=pat.str[0].eq('D'), ended=pat.str[-1].map({'D': 'ended DNP', 'L': 'ended limited',
                                                                          'F': 'ended full'}).fillna('other'))
    rows = []
    base = live[live['status'] == 'unlisted']
    f0 = {pos: _factor(g['actual_pts'], g['model_pts']) for pos, g in base.groupby('Pos')}
    for col in ('early_dnp', 'ended'):
        for val, g in lq.groupby(col):
            rel = g['actual_pts'] / g['Pos'].map(f0)
            rows.append({'split': f'{col}={val}', 'rows': len(g), 'factor': _factor(rel, g['model_pts']),
                         'pts bias': float((g['actual_pts'] - g['model_pts']).mean())})
    sections += [pd.DataFrame(rows).to_string(index=False, float_format=fmt)]
    text = '\n'.join(sections)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
