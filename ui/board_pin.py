"""
A built weekly board kept on disk, and a ruling that re-projects only its team (2026-10-23, user: "if saving board to disk
helps keep it, and add in the team only reload"; ported from CFB Scholar's ui/board_pin.py).

  - a board is saved to data/cache/boards/ (git-ignored) when it is built, so a new browser session or an app restart picks
    it up again for BOARD_KEEP_HOURS, as long as the model code and default features are unchanged (model_signature);
  - an injury/availability ruling, a FantasyPros injury pull or a QB1 choice re-projects only the teams whose rows changed
    (build_weekly_projections' only_teams) and swaps their rows, explanations and ledgers into the board (merge_team_board).

Nothing here changes a projection: a merged board is the board a full build would give (tests/test_board_pin.py).
"""
import datetime
import glob
import hashlib
import os
import pickle
import time

import pandas as pd

from data.availability_overrides import AVAILABILITY_OVERRIDE_PATH, load_availability_overrides
from data.fantasypros_availability import FANTASYPROS_INJURY_PATH

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, 'data', 'cache', 'boards')
BOARD_KEEP_HOURS = 12.0
MAX_SAVED_BOARDS = 4
# the ledgers in a board's meta, each a list of {'team': ...} entries
TEAM_LEDGERS = ('vacancy_ledger', 'rb_allocation_ledger', 'pass_capacity_ledger', 'rb_carry_budget_ledger')


def model_signature():
    """Changes when the model's code or default features change, so a saved board from older code is not shown."""
    from data.weekly_projections import DEFAULT_FEATURES
    stamps = sorted((os.path.basename(p), int(os.path.getmtime(p))) for p in glob.glob(os.path.join(REPO, 'data', '*.py')))
    return hashlib.sha1(repr((stamps, sorted(DEFAULT_FEATURES))).encode('utf-8')).hexdigest()[:16]


def ruling_snapshot(year, week):
    """{team: frozenset of that team's rows} over the ruling sources for this (year, week): your availability
    overrides, the FantasyPros injury pull, the season's QB1 choices, and (2026-10-25, v2_ps_line_routing) the team's
    RBs with a posted line this week."""
    from data.weekly_projections import load_qb1_overrides
    frames = []
    for path in (AVAILABILITY_OVERRIDE_PATH, FANTASYPROS_INJURY_PATH):
        try:
            rows, _problem = load_availability_overrides(year, week, path=path)
        except Exception:
            rows = None
        frames.append(('avail', rows))
    try:
        qb1, _problem = load_qb1_overrides(year)
    except Exception:
        qb1 = None
    frames.append(('qb1', qb1))
    out = {}
    for source, frame in frames:
        if frame is None or frame.empty or 'team' not in frame.columns:
            continue
        for rec in frame.astype(str).to_dict('records'):
            team = str(rec.get('team', '')).strip().upper()
            out.setdefault(team, set()).add((source,) + tuple(sorted(rec.items())))
    try:
        from data.odds_weekly import rb_lines_by_team
        lines = rb_lines_by_team(year, week)
    except Exception:
        lines = {}
    for team, names in lines.items():
        out.setdefault(team, set()).update(('lines', name) for name in names)
    return {team: frozenset(rows) for team, rows in out.items()}


def changed_teams(old, new):
    """Teams whose ruling rows differ between two snapshots; None when a changed row names no team (rebuild in full)."""
    if old is None:
        return None
    teams = {t for t in set(old) | set(new) if old.get(t, frozenset()) != new.get(t, frozenset())}
    if any(not str(t).strip() or str(t).lower() == 'nan' for t in teams):
        return None
    return teams


def merge_team_board(df, meta, part, part_meta, teams):
    """(board, meta) with `teams` re-projected: their rows, explanations and ledger entries replaced by `part`'s, then the
    positional ranks, boom/bust bands and chart axes recomputed on the merged board, as a full build does at its end."""
    from data.weekly_projections import board_distribution, set_distribution_axes
    teams = {str(t).upper() for t in teams}
    on_team = df['Team'].astype(str).str.upper().isin(teams)
    fresh = part[part['Team'].astype(str).str.upper().isin(teams)] if part is not None and not part.empty else part
    merged = pd.concat([df[~on_team], fresh], ignore_index=True, sort=False)
    merged = merged[list(df.columns) + [c for c in merged.columns if c not in df.columns]]
    merged = merged.sort_values('Model Proj Pts', ascending=False, kind='mergesort').reset_index(drop=True)

    meta = dict(meta or {})
    part_meta = part_meta or {}
    explanations = {k: v for k, v in (meta.get('explanations') or {}).items() if str(k[2]).upper() not in teams}
    explanations.update({k: v for k, v in (part_meta.get('explanations') or {}).items() if str(k[2]).upper() in teams})
    for ledger in TEAM_LEDGERS:
        kept = [e for e in (meta.get(ledger) or []) if str(e.get('team', '')).upper() not in teams]
        meta[ledger] = kept + [e for e in (part_meta.get(ledger) or []) if str(e.get('team', '')).upper() in teams]
    rank = merged.groupby('Pos')['Model Proj Pts'].rank(ascending=False, method='min')
    for idx, row in merged.iterrows():
        detail = explanations.get((row['Player'], row['Pos'], row['Team']))
        if detail is not None:
            detail['distribution'] = board_distribution(row, rank.loc[idx], detail)
    set_distribution_axes(explanations)
    meta['explanations'] = explanations
    meta['players'] = int(len(merged))
    meta['vacancy_adjusted'] = bool(meta.get('vacancy_adjusted')) or bool(part_meta.get('vacancy_adjusted'))
    return merged, meta


def built_label(built_at):
    """'Fri 2:42 PM' for a naive local build time."""
    return built_at.strftime('%a %I:%M %p').replace(' 0', ' ')


def _path(key):
    year, week, scoring = key
    return os.path.join(CACHE_DIR, f"board_{year}_wk{int(week):02d}_{str(scoring).replace(' ', '').replace('-', '')}.pkl")


def save(key, entry):
    """Write the entry to disk (never raises: a failed save only means the next session builds again). Keeps the newest
    MAX_SAVED_BOARDS files."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _path(key) + '.tmp'
        with open(tmp, 'wb') as handle:
            pickle.dump({**entry, 'signature': model_signature()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _path(key))
        saved = sorted(glob.glob(os.path.join(CACHE_DIR, 'board_*.pkl')), key=os.path.getmtime, reverse=True)
        for old in saved[MAX_SAVED_BOARDS:]:
            os.remove(old)
    except Exception:
        pass


def load(key, max_age_hours=BOARD_KEEP_HOURS):
    """The saved entry for this key when it is younger than `max_age_hours` and from the current model code, else None."""
    path = _path(key)
    try:
        if not os.path.exists(path) or (time.time() - os.path.getmtime(path)) > max_age_hours * 3600:
            return None
        with open(path, 'rb') as handle:
            entry = pickle.load(handle)
        if entry.get('signature') != model_signature():
            return None
        entry.pop('signature', None)
        return entry
    except Exception:
        return None


def has_saved(key, max_age_hours=BOARD_KEEP_HOURS):
    path = _path(key)
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) <= max_age_hours * 3600


def age_minutes(built_at, now=None):
    now = now or datetime.datetime.now()
    return max(0, int((now - built_at).total_seconds() // 60))
