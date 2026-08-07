"""
Import a FantasyPros cheat sheet into this app's board.

WHAT THIS IS FOR: their cheat-sheet payload carries three things this app
cannot derive on its own, and one of them unlocks a draft format the board
couldn't price at all.

  Auction values    - dollar values under a $200 budget. VORP says who is
                      worth more; it cannot say whether that is a $47 player
                      or a $12 one, and an auction draft is entirely that
                      question. The board has always offered "Auction" as a
                      draft type with nothing behind it.
  Expert tiers      - drawn by their analysts. This app already computes
                      tiers by k-means on projected points, which is a
                      different and equally valid answer; carrying both means
                      the disagreement is visible, which is the useful part.
  Expert notes      - a written paragraph per player on why he's ranked
                      where he is.

WHAT THIS DELIBERATELY IS NOT: a scraper. Nothing here talks to their
servers, polls their API or automates a login. It reads a file you already
have, once, when you hand it over - the same posture as data/ffa_import.py.

THE ID PROBLEM. Their cheat sheet identifies players by FantasyPros ID and
carries no names at all. Two ways to resolve them, tried in order: the
player array their draft room ships alongside it (100% coverage, and it's in
the same capture folder), then this app's existing crosswalk, whose
fantasypros_id column covers about 80% of the sheet. Anything still
unresolved is dropped rather than guessed - a mis-keyed auction value is
worse than a missing one.
"""
import json
import os
import re

import numpy as np
import pandas as pd

# Persisted next to the FFA import, and gitignored for the same reason: this
# is subscription content that belongs on your machine, not in a repo.
CHEATSHEET_PATH = os.path.join('external_data', 'fantasypros_cheatsheet.json')
PLAYER_ARRAY_PATH = os.path.join('external_data', 'fantasypros_players.json')

# Columns this import contributes to the board.
CHEATSHEET_COLUMNS = ['Auction $', 'FP Tier', 'FP Note']


def _player_array_from_text(text):
    """
    The id -> name/team/position map out of their draft room's player array.

    Ships as `window.PLAYER_DATA = {...};` inside an HTML/JS response, so the
    assignment is stripped before parsing rather than requiring the caller to
    have cleaned it up.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode('utf-8', errors='replace')
    text = str(text).strip()
    marker = 'PLAYER_DATA'
    if marker in text:
        text = text.split(marker, 1)[1]
        text = text[text.index('{'):] if '{' in text else text
    text = text.rstrip().rstrip(';')
    try:
        payload = json.loads(text)
    except Exception:
        # Trailing script tags or markup after the object.
        end = text.rfind('}')
        if end < 0:
            return {}
        try:
            payload = json.loads(text[:end + 1])
        except Exception:
            return {}
    players = payload.get('players') if isinstance(payload, dict) else payload
    if not isinstance(players, list):
        return {}
    out = {}
    for entry in players:
        if not isinstance(entry, dict) or entry.get('id') is None:
            continue
        out[str(entry['id'])] = {
            'Player': str(entry.get('name') or '').strip(),
            'Pos': str(entry.get('position') or entry.get('eligibility') or '').upper().strip(),
            'Team': str(entry.get('team') or '').upper().strip(),
            'Bye': entry.get('bye_week'),
        }
    return {k: v for k, v in out.items() if v['Player']}


def _clean_note(note):
    """
    Their notes end with an attribution carrying a raw <a> tag. Rendered as
    text it shows the markup; rendered as markdown it would be a live link
    out of the draft room mid-pick. Tags are stripped and the author's name
    kept, since knowing which analyst wrote it is part of reading it.
    """
    if not isinstance(note, str) or not note.strip():
        return None
    text = re.sub(r'<[^>]+>', '', note)
    text = text.replace('&amp;', '&').replace('&nbsp;', ' ').replace('&#39;', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text or None


def _crosswalk_names():
    """id -> name/team/position from this app's existing player crosswalk."""
    from data.draft_sources import load_player_id_map
    ids, _ = load_player_id_map()
    if ids is None or ids.empty or 'fantasypros_id' not in ids.columns:
        return {}
    keyed = ids.dropna(subset=['fantasypros_id'])
    out = {}
    for _, row in keyed.iterrows():
        try:
            key = str(int(float(row['fantasypros_id'])))
        except (TypeError, ValueError):
            continue
        out.setdefault(key, {
            'Player': str(row.get('name') or '').strip(),
            'Pos': str(row.get('position') or '').upper().strip(),
            'Team': str(row.get('team') or '').upper().strip(),
            'Bye': np.nan,
        })
    return out


def parse_cheatsheet(raw, player_array=None):
    """
    Their cheat-sheet payload into a tidy frame.

    `raw` is the parsed getCheatSheet JSON; `player_array` is the optional
    id->name map from _player_array_from_text. Returns (df, error).
    """
    groups = raw.get('rankings') if isinstance(raw, dict) else None
    if not isinstance(groups, list) or not groups:
        return pd.DataFrame(), ("that JSON has no 'rankings' array — you want the "
                                "getCheatSheet response from a FantasyPros draft page.")

    names = dict(player_array or {})
    if not names:
        names = _crosswalk_names()
    if not names:
        return pd.DataFrame(), ("couldn't resolve player IDs to names: the crosswalk is "
                                "unavailable and no player-array file was supplied.")

    # The overall group is preferred where a player appears in several, since
    # its tiers are drawn across positions.
    rows, seen = [], set()
    ordered = sorted(groups, key=lambda g: 0 if str(g.get('position', '')).lower() == 'overall' else 1)
    for group in ordered:
        for entry in (group.get('players') or []):
            key = str(entry.get('id'))
            if key in seen or key not in names:
                continue
            seen.add(key)
            identity = names[key]
            auction = pd.to_numeric(entry.get('auctionValue'), errors='coerce')
            tier = pd.to_numeric(entry.get('tier'), errors='coerce')
            note = entry.get('expertNote')
            rows.append({
                'Player': identity['Player'],
                'Pos': identity['Pos'],
                'Team': identity['Team'],
                'Auction $': float(auction) if pd.notna(auction) else np.nan,
                'FP Tier': int(tier) if pd.notna(tier) else pd.NA,
                'FP Note': _clean_note(note),
            })

    if not rows:
        return pd.DataFrame(), "no players in that cheat sheet could be matched to a name."
    df = pd.DataFrame(rows)
    df['FP Tier'] = df['FP Tier'].astype('Int64')
    return df, None


def load_cheatsheet(path=CHEATSHEET_PATH, array_path=PLAYER_ARRAY_PATH):
    """Read a persisted cheat sheet from disk. Returns (df, error)."""
    if not os.path.exists(path):
        return pd.DataFrame(), None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            raw = json.load(handle)
    except Exception as exc:
        return pd.DataFrame(), f"couldn't read {path}: {exc}"
    array = {}
    if os.path.exists(array_path):
        try:
            with open(array_path, 'r', encoding='utf-8') as handle:
                array = _player_array_from_text(handle.read())
        except Exception:
            array = {}
    return parse_cheatsheet(raw, array)


def save_upload(uploaded_file):
    """
    Persist whichever of the two files this is, auto-detected.

    One uploader rather than two, because the two files come out of the same
    capture folder and asking someone to work out which slot each belongs in
    is a needless way to get it wrong. Returns (kind, error).
    """
    if uploaded_file is None:
        return None, None
    try:
        uploaded_file.seek(0)
        blob = uploaded_file.read()
    except Exception as exc:
        return None, f"couldn't read that file: {exc}"

    text = blob.decode('utf-8', errors='replace') if isinstance(blob, (bytes, bytearray)) else str(blob)
    os.makedirs(os.path.dirname(CHEATSHEET_PATH), exist_ok=True)

    if 'PLAYER_DATA' in text or '"eligibility"' in text:
        if not _player_array_from_text(text):
            return None, "that looks like a player array but no players could be read from it."
        with open(PLAYER_ARRAY_PATH, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return 'player array', None

    try:
        raw = json.loads(text)
    except Exception as exc:
        return None, f"not valid JSON: {exc}"
    if not (isinstance(raw, dict) and isinstance(raw.get('rankings'), list)):
        return None, ("unrecognised file — expected the getCheatSheet response (a JSON object "
                      "with a 'rankings' array) or the draft room's player array.")
    with open(CHEATSHEET_PATH, 'w', encoding='utf-8') as handle:
        json.dump(raw, handle)
    return 'cheat sheet', None


def merge_cheatsheet_into_board(board, sheet):
    """
    Fold auction values, expert tiers and notes onto the board.

    Matched with the app's usual two-tier name key. Nothing is overwritten:
    the board's own Tier column stays exactly as computed, and FP Tier sits
    beside it, because the two disagreeing is information rather than a
    conflict to resolve.
    """
    if board is None or board.empty or sheet is None or sheet.empty:
        return board, {'matched': 0}

    from data.utils import clean_name_exact, clean_name_for_merge

    out = board.copy()
    keyed = sheet.copy()
    keyed['_exact'] = clean_name_exact(keyed['Player'])
    keyed['_loose'] = clean_name_for_merge(keyed['Player'])
    exact = {k: i for i, k in enumerate(keyed['_exact'])}
    loose = {k: i for i, k in enumerate(keyed['_loose'])}

    board_exact = clean_name_exact(out['Player'])
    board_loose = clean_name_for_merge(out['Player'])
    positions = []
    for i in range(len(out)):
        index = exact.get(board_exact.iloc[i])
        if index is None:
            index = loose.get(board_loose.iloc[i])
        positions.append(index)

    for column in CHEATSHEET_COLUMNS:
        if column not in keyed.columns:
            continue
        values = [keyed[column].iloc[p] if p is not None else None for p in positions]
        out[column] = values
    if 'Auction $' in out.columns:
        out['Auction $'] = pd.to_numeric(out['Auction $'], errors='coerce')
    if 'FP Tier' in out.columns:
        out['FP Tier'] = pd.to_numeric(out['FP Tier'], errors='coerce').astype('Int64')
    return out, {'matched': int(sum(1 for p in positions if p is not None)),
                 'imported': int(len(sheet))}
