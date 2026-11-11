"""
Weekly PFF archive coverage and labelling check (2026-11-11, audit D4, docs/audit_leads_2026-11-10.md). Read only.

For every season with a pff_imports/<year>/weekly/ folder: which week folders hold a complete receiving_summary + receiving_concept
pair (what data/pff_alignment.py reads) and a receiving_scheme, and whether each folder's export really is the week its folder
says. A weekly export lists the players of the teams that played that week, so the teams in receiving_summary.csv are matched
against each regular-season week's schedule: a team on bye cannot appear. A folder whose best-matching week is not its own is
mislabelled. The loader reads folders strictly before the projected week by folder number, so a week saved one folder too high
is invisible to the next board and read one week late after that.

It also prints the manifest rows that are missing or disagree with the folders, and the D4 gate: retry backlog N2 / N5 / C1 /
C1b once three seasons have a full regular season of weekly exports.

    python scripts/pff_weekly_coverage.py [--years 2024,2025,2026] [--root pff_imports]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import pandas as pd  # noqa: E402

from config import pff_team_to_abbr  # noqa: E402
from data.pff_alignment import detect_export_week  # noqa: E402

REGULAR_WEEKS = range(1, 19)
FULL_SEASON_WEEKS = 17          # weeks 1-17 (week 18 is out of scope)


def week_teams(schedule):
    """{week: set of teams with a regular-season game that week}."""
    s = schedule[schedule['game_type'].eq('REG')] if 'game_type' in schedule.columns else schedule
    out = {}
    for week, home, away in zip(s['week'], s['home_team'], s['away_team']):
        out.setdefault(int(week), set()).update({str(home), str(away)})
    return out


def best_week(teams, by_week):
    """(week, score): the regular-season week whose playing teams best match `teams` (Jaccard), or (None, 0)."""
    best, score = None, 0.0
    for week, playing in by_week.items():
        j = len(teams & playing) / len(teams | playing) if teams | playing else 0.0
        if j > score + 1e-9:
            best, score = week, j
    return best, score


def box_scores(year):
    """(weekly stats, name column) for the season, or (None, None) when they cannot be loaded."""
    try:
        from data.transforms import load_and_merge_data
        stats, _t, name_col, _ = load_and_merge_data(year, 'Full PPR')
    except Exception:
        return None, None
    return stats[pd.to_numeric(stats['week'], errors='coerce').isin(REGULAR_WEEKS)], name_col


def folder_report(year, weekly_dir, by_week, box=(None, None)):
    """One row per week folder. The week is read from the stats when the box scores are loaded (a folder's PFF receiving yards
    match one week's box scores almost exactly and no other week's - checked on 2026, ~100% against ~5%), else from the teams."""
    rows = []
    for name in sorted(os.listdir(weekly_dir), key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0)):
        path = os.path.join(weekly_dir, name)
        if not (os.path.isdir(path) and name.isdigit()):
            continue
        week = int(name)
        files = set(os.listdir(path))
        pair = {'receiving_summary.csv', 'receiving_concept.csv'} <= files
        teams, n_players, summary = set(), 0, None
        if 'receiving_summary.csv' in files:
            try:
                summary = pd.read_csv(os.path.join(path, 'receiving_summary.csv'), usecols=['player', 'team_name', 'yards'])
                teams = {pff_team_to_abbr(str(t)) for t in summary['team_name'].dropna()}
                n_players = len(summary)
            except (OSError, ValueError):
                summary = None
        stat_week = (detect_export_week(summary, box[0], box[1]) if (summary is not None and box[0] is not None and week in REGULAR_WEEKS)
                     else (None, 0.0, 0))
        if stat_week[0] is not None and stat_week[1] >= 0.8:
            matched, score = stat_week[0], stat_week[1]
            verdict = 'ok (stats)' if matched == week else f'looks like week {matched} (stats)'
        elif week in REGULAR_WEEKS and teams:
            matched, score = best_week(teams, by_week)
            playing = by_week.get(week, set())
            own = len(teams & playing) / len(teams | playing) if teams | playing else 0.0
            ties = sum(1 for p in by_week.values() if abs(len(teams & p) / len(teams | p) - score) < 1e-9)
            if own >= score - 1e-9:
                # a week with no byes matches every other no-bye week equally: the teams cannot tell them apart
                matched, score = week, own
                verdict = 'ok' if ties == 1 else 'ok (no byes: unverifiable by teams)'
            else:
                verdict = f'looks like week {matched}' if matched else 'no match'
        elif week in REGULAR_WEEKS:
            matched, score, verdict = None, 0.0, 'empty' if not files else 'no summary'
        else:
            matched, score, verdict = None, 0.0, 'postseason folder'
        rows.append({'year': year, 'folder': week, 'pair': pair, 'scheme': 'receiving_scheme.csv' in files, 'players': n_players,
                     'teams': len(teams), 'best_week': matched, 'match': round(score, 2), 'verdict': verdict})
    return pd.DataFrame(rows)


def week_by_date(schedule):
    """[(last gameday, week)] for the regular season, in week order."""
    s = schedule[schedule['game_type'].eq('REG')] if 'game_type' in schedule.columns else schedule
    last = pd.to_datetime(s['gameday']).groupby(s['week'].astype(int)).max()
    return list(zip(last.to_numpy(), last.index))


def manifest_report(weekly_dir, folders, schedule=None):
    path = os.path.join(weekly_dir, 'manifest.csv')
    if not os.path.exists(path):
        return ['  no manifest.csv']
    m = pd.read_csv(path)
    rows = set(pd.to_numeric(m['week'], errors='coerce').dropna().astype(int))
    have = set(folders.loc[folders['pair'], 'folder'])
    notes = []
    if schedule is not None and 'export_date' in m.columns:
        # an in-season export covers the last week finished before it; a backfill (exported after the season) is not checked
        ends = week_by_date(schedule)
        for week, date in zip(pd.to_numeric(m['week'], errors='coerce'), pd.to_datetime(m['export_date'], errors='coerce')):
            if pd.isna(week) or pd.isna(date) or not ends or date > ends[-1][0] + pd.Timedelta(days=14):
                continue
            done = [w for end, w in ends if end < date]
            implied = done[-1] if done else None
            if implied is not None and int(week) <= ends[-1][1] and implied != int(week):
                notes.append(f'  manifest row {int(week)}: exported {date:%Y-%m-%d}, after week {implied} finished')
    if have - rows:
        notes.append(f'  folders with a pair but no manifest row: {sorted(have - rows)}')
    if rows - set(folders['folder']):
        notes.append(f'  manifest rows with no folder: {sorted(rows - set(folders["folder"]))}')
    empty = set(folders.loc[~folders['pair'], 'folder']) & rows
    if empty:
        notes.append(f'  manifest rows whose folder has no pair: {sorted(empty)}')
    return notes or ['  manifest matches the folders']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default=None, help='comma-separated; default every season with a weekly folder')
    ap.add_argument('--root', default='pff_imports')
    a = ap.parse_args()
    from data.loaders import load_schedule
    years = ([int(y) for y in a.years.split(',')] if a.years else
             sorted(int(y) for y in os.listdir(a.root) if y.isdigit() and os.path.isdir(os.path.join(a.root, y, 'weekly'))))
    complete_seasons = []
    for year in years:
        weekly_dir = os.path.join(a.root, str(year), 'weekly')
        if not os.path.isdir(weekly_dir):
            print(f'{year}: no weekly folder')
            continue
        schedule = load_schedule(year)
        folders = folder_report(year, weekly_dir, week_teams(schedule), box_scores(year))
        reg = folders[folders['folder'].isin(REGULAR_WEEKS)]
        good = reg[reg['pair'] & reg['verdict'].str.startswith('ok')]
        print(f'\n{year}: {len(good)} regular-season weeks with a complete, correctly labelled pair '
              f'(scheme on {int(reg["scheme"].sum())})')
        print(folders.to_string(index=False))
        bad = reg[~reg['verdict'].str.startswith('ok')]
        for r in bad.itertuples():
            print(f'  FOLDER {r.folder}: {r.verdict}' + (f' (match {r.match})' if r.best_week else ''))
        held = reg.dropna(subset=['best_week']).groupby('best_week')['folder'].apply(list)
        for wk, fl in held[held.apply(len) > 1].items():
            print(f'  WEEK {int(wk)} is in folders {fl}')
        found = set(reg['best_week'].dropna().astype(int))
        missing = [w for w in range(1, max(found, default=0) + 1) if w not in found]   # up to the latest week on disk
        if missing:
            print(f'  weeks with no folder holding them: {missing}')
        print('\n'.join(manifest_report(weekly_dir, folders, schedule)))
        if set(range(1, FULL_SEASON_WEEKS + 1)) <= set(good['folder']):
            complete_seasons.append(year)
    legacy = [y for y in sorted(os.listdir(a.root)) if os.path.isdir(os.path.join(a.root, y, 'Weekly Data'))]
    if legacy:
        print(f'\nNot in the loader layout (pff_imports/<year>/Weekly Data, numbered files): {legacy}')
    print(f'\nD4 gate: full weekly seasons {complete_seasons} ({len(complete_seasons)} of the 3 needed to retry N2 / N5 / C1 / C1b)')


if __name__ == '__main__':
    main()
