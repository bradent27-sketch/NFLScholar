"""
Per-game box scores, and the resolver that gets from any per-game stat in
the app back to the game it came from.

WHAT THE SOURCE SURVEY FOUND, because it decided the whole design (the
porting notes say measure before choosing, and here it deleted two entire
subsystems the basketball version needs):

  `stats_player_week_{season}.csv` - already on disk, already downloaded,
  already loaded by six other tabs - carries `game_id`, and it is THE SAME
  ID the nflverse schedule uses. Measured on the real 2025 files: 285
  distinct games in the weekly stats, 285 in the schedule, **285/285 = 100%
  match in both directions**, with a median of 68 player rows per game.

  Both sides store it as a plain string, so there is no int32-vs-str cast
  hazard here - the bug that returned a silently empty box in the CBB port.
  It's cast defensively anyway, because that failure is invisible.

Two consequences:

  THE BOX SCORE COSTS NOTHING TO ADD. No new file, no new download, no API
  key. It works offline on a fresh clone for every season 2019-2025.

  THERE IS NO SECOND ID NAMESPACE. The basketball port needs a date+team
  fallback because CBBD's game ids are unrelated to ESPN's. Here one id
  serves everything, and the fallback path below keys on (week, team) -
  which in the NFL is genuinely unique, since a team plays exactly one game
  per week. That's a stronger fallback than the sport this was ported from
  can have, not a weaker one.

TEAM TOTALS: what may be summed from player rows, and what may not.

  The basketball notes are emphatic that team totals must NOT be summed
  from player rows, because team rebounds and team turnovers belong to no
  player. Football mostly doesn't have that problem - the weekly export
  carries penalties, fumbles and every defensive stat per player - so the
  sums here are exact for what's shown.

  The real trap is different, and it's a double-count: a completed pass
  produces a `passing_first_downs` on the QUARTERBACK and a
  `receiving_first_downs` on the RECEIVER. Summing every *_first_downs
  column counts each passing first down twice. See _team_totals.

  Time of possession, third-down efficiency and total plays are genuinely
  not derivable from player rows, so they are NOT shown at all rather than
  approximated. The panel says so, so numbers not tying out against a TV
  box score doesn't read as a bug.
"""
import pandas as pd

from data.utils import get_val  # noqa: F401  (kept for callers importing from here)

# Column groups per box-score section. Each entry is (column, header).
PASSING_COLS = [
    ('passing_completions', 'C'), ('passing_attempts', 'A'), ('passing_yards', 'YDS'),
    ('passing_tds', 'TD'), ('passing_interceptions', 'INT'), ('sacks_suffered', 'SCK'),
]
RUSHING_COLS = [
    ('rushing_attempts', 'CAR'), ('rushing_yards', 'YDS'), ('rushing_tds', 'TD'),
    ('rushing_first_downs', 'FD'),
]
RECEIVING_COLS = [
    ('targets', 'TGT'), ('receptions', 'REC'), ('receiving_yards', 'YDS'),
    ('receiving_tds', 'TD'), ('receiving_first_downs', 'FD'),
]
DEFENSE_COLS = [
    ('def_tackles_solo', 'SOLO'), ('def_tackles_with_assist', 'AST'), ('def_sacks', 'SCK'),
    ('def_tackles_for_loss', 'TFL'), ('def_qb_hits', 'QBH'), ('def_interceptions', 'INT'),
    ('def_pass_defended', 'PD'), ('def_fumbles_forced', 'FF'),
]
KICKING_COLS = [
    ('fg_made', 'FGM'), ('fg_att', 'FGA'), ('fg_long', 'LNG'),
    ('pat_made', 'XPM'), ('pat_att', 'XPA'),
]

BOX_SECTIONS = [
    ('Passing', PASSING_COLS, 'passing_attempts'),
    ('Rushing', RUSHING_COLS, 'rushing_attempts'),
    ('Receiving', RECEIVING_COLS, 'targets'),
    ('Defense', DEFENSE_COLS, None),
    ('Kicking', KICKING_COLS, 'fg_att'),
]


def _numeric(frame, col):
    if col not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype='float64')
    return pd.to_numeric(frame[col], errors='coerce').fillna(0.0)


def game_players(stats_df, game_id, id_col='game_id'):
    """
    Every player row for one game.

    `game_id` is cast to string on BOTH sides. The two sources agree on the
    type today (both str), but this comparison failing returns an empty box
    with no error anywhere - the single most expensive bug in the port this
    was translated from - so it is not left to chance.
    """
    if stats_df is None or stats_df.empty or id_col not in stats_df.columns:
        return pd.DataFrame()
    return stats_df[stats_df[id_col].astype(str) == str(game_id)].copy()


def section_rows(players, team, columns, gate_col, name_col='player_display_name'):
    """
    One box-score section for one team: the players with something to show,
    ordered by the section's headline stat.

    `gate_col` is what decides "has something to show" - a receiver with no
    targets doesn't belong in the receiving table. Passing `gate_col=None`
    (defense) instead includes anyone with a non-zero value in ANY of the
    section's columns, because there is no single stat a defender must have.
    """
    if players.empty:
        return pd.DataFrame()
    rows = players[players['team'].astype(str).str.upper() == str(team).upper()]
    if rows.empty:
        return pd.DataFrame()
    present = [(col, header) for col, header in columns if col in rows.columns]
    if not present:
        return pd.DataFrame()

    if gate_col and gate_col in rows.columns:
        keep = _numeric(rows, gate_col) > 0
    else:
        keep = pd.Series(False, index=rows.index)
        for col, _ in present:
            keep = keep | (_numeric(rows, col) > 0)
    rows = rows[keep]
    if rows.empty:
        return pd.DataFrame()

    out = pd.DataFrame({'Player': rows[name_col].astype(str)})
    out['Pos'] = rows['position'].astype(str) if 'position' in rows.columns else ''
    for col, header in present:
        out[header] = _numeric(rows, col)
    sort_header = next((h for c, h in present if c == gate_col), present[0][1])
    return out.sort_values(sort_header, ascending=False).reset_index(drop=True)


def team_totals(players, team):
    """
    Team totals that are EXACT as a sum of player rows. Nothing here is an
    approximation, and anything that would be one is absent.

    First downs deliberately exclude `receiving_first_downs`: a completed
    pass credits the quarterback with a passing first down AND the receiver
    with a receiving first down, so adding both double-counts every one of
    them. Passing + rushing is the correct total.

    Turnovers are interceptions THROWN plus fumbles LOST - not fumbles
    committed, since a fumble a team recovers itself is not a turnover.
    """
    if players.empty:
        return {}
    rows = players[players['team'].astype(str).str.upper() == str(team).upper()]
    if rows.empty:
        return {}

    passing = float(_numeric(rows, 'passing_yards').sum())
    rushing = float(_numeric(rows, 'rushing_yards').sum())
    interceptions = float(_numeric(rows, 'passing_interceptions').sum())
    # 'sack_fumbles_lost'/'rushing_fumbles_lost'/'receiving_fumbles_lost' are
    # the three lost-fumble columns the weekly export carries; a plain
    # 'fumbles_lost_total' also exists and is preferred when present.
    if 'fumbles_lost_total' in rows.columns:
        fumbles_lost = float(_numeric(rows, 'fumbles_lost_total').sum())
    else:
        fumbles_lost = float(sum(
            _numeric(rows, c).sum()
            for c in ('sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost')
        ))
    return {
        'Pass Yds': passing,
        'Rush Yds': rushing,
        'Total Yds': passing + rushing,
        'First Downs': float(_numeric(rows, 'passing_first_downs').sum()
                             + _numeric(rows, 'rushing_first_downs').sum()),
        'Turnovers': interceptions + fumbles_lost,
        'Sacks Allowed': float(_numeric(rows, 'sacks_suffered').sum()),
        'Penalties': float(_numeric(rows, 'penalties').sum()),
        'Penalty Yds': float(_numeric(rows, 'penalty_yards').sum()),
    }


# Which totals are compared as a shared split bar, and whether more is
# better for the team that has more of it. A turnover count split evenly
# would otherwise imply the team with more turnovers is winning that row.
TOTALS_COMPARISON = [
    ('Total Yds', True), ('Pass Yds', True), ('Rush Yds', True),
    ('First Downs', True), ('Turnovers', False), ('Sacks Allowed', False),
    ('Penalty Yds', False),
]


def leading_performers(players, team, name_col='player_display_name'):
    """
    The one line worth reading first per phase - passer, rusher, receiver -
    picked by yards. Returns [] rather than zero-yard entries for a team
    that genuinely did nothing in a phase.
    """
    if players.empty:
        return []
    rows = players[players['team'].astype(str).str.upper() == str(team).upper()]
    if rows.empty:
        return []
    out = []
    for label, yards_col, detail in (
        ('Passing', 'passing_yards',
         lambda r: f"{int(_v(r, 'passing_completions'))}/{int(_v(r, 'passing_attempts'))}, "
                   f"{int(_v(r, 'passing_yards'))} yds, {int(_v(r, 'passing_tds'))} TD"),
        ('Rushing', 'rushing_yards',
         lambda r: f"{int(_v(r, 'rushing_attempts'))} car, {int(_v(r, 'rushing_yards'))} yds, "
                   f"{int(_v(r, 'rushing_tds'))} TD"),
        ('Receiving', 'receiving_yards',
         lambda r: f"{int(_v(r, 'receptions'))} rec, {int(_v(r, 'receiving_yards'))} yds, "
                   f"{int(_v(r, 'receiving_tds'))} TD"),
    ):
        if yards_col not in rows.columns:
            continue
        values = _numeric(rows, yards_col)
        if values.max() <= 0:
            continue
        best = rows.loc[values.idxmax()]
        out.append({'phase': label, 'player': str(best[name_col]), 'line': detail(best)})
    return out


def _v(row, col):
    try:
        value = float(row.get(col, 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(value) else value


# ---------------------------------------------------------------------------
# The resolver: any per-game stat -> the game it came from
# ---------------------------------------------------------------------------

def game_link_rows(log_df, slate_df, team=None, id_col='game_id', week_col='week', limit=None):
    """
    One entry per row of `log_df` that resolves to a real game on `slate_df`,
    in the log's own order: {'game_id', 'week', 'label', 'help', 'won',
    'opponent'}.

    TWO RESOLUTION PATHS, tried in order:

      1. By `game_id`, when the log carries one - which everything sourced
         from the weekly stats export does.
      2. By (week, team), for a log that only knows when a game happened -
         a computed/aggregated frame, or a source without ids. In the NFL
         this is exact: a team plays exactly one game per week, so unlike
         the date+team fallback this was ported from, it cannot be
         ambiguous. It DOES require `team`, and returns nothing without it.

    A ROW THAT RESOLVES TO NOTHING IS DROPPED, never emitted with a null id.
    A dead link that navigates nowhere is worse than no link at all.

    Labels use `@ABBR` for an away game and `vs ABBR` for a home game -
    with the space, because `vsOAK` is unreadable. A neutral-site game is
    never labelled as an away game.
    """
    empty = []
    if log_df is None or len(log_df) == 0 or slate_df is None or slate_df.empty:
        return empty
    if 'Game Id' not in slate_df.columns:
        return empty

    by_id = {str(r['Game Id']): r for _, r in slate_df.iterrows()}
    by_week = {}
    if team:
        team_up = str(team).upper()
        for _, r in slate_df.iterrows():
            if team_up in (str(r['Away']).upper(), str(r['Home']).upper()):
                by_week[int(r['Week'])] = r

    seen, out = set(), []
    for _, row in log_df.iterrows():
        game = None
        raw_id = row.get(id_col) if id_col in log_df.columns else None
        if raw_id is not None and pd.notna(raw_id):
            game = by_id.get(str(raw_id))
        if game is None and week_col in log_df.columns:
            week = row.get(week_col)
            if pd.notna(week):
                try:
                    game = by_week.get(int(week))
                except (TypeError, ValueError):
                    game = None
        if game is None:
            continue
        entry = _link_entry(game, team)
        if entry['game_id'] in seen:
            continue
        seen.add(entry['game_id'])
        out.append(entry)
    # `limit` keeps the MOST RECENT, which is what a strip under a game log
    # wants - the last five games, not the first five.
    return out[-limit:] if limit else out


def _link_entry(game, team):
    away, home = str(game['Away']).upper(), str(game['Home']).upper()
    subject = str(team).upper() if team else away
    is_home = subject == home
    opponent = away if is_home else home
    neutral = bool(game.get('Neutral Site'))
    # A neutral-site game is never an away game - "@" would be wrong.
    prefix = 'vs ' if (is_home or neutral) else '@'
    won = None
    winner = game.get('Winner')
    if winner is not None and pd.notna(winner) and bool(game.get('Played')):
        won = str(winner).upper() == subject

    score = ''
    if game.get('Played') and pd.notna(game.get('Away Pts')) and pd.notna(game.get('Home Pts')):
        mine = game['Home Pts'] if is_home else game['Away Pts']
        theirs = game['Away Pts'] if is_home else game['Home Pts']
        score = f" {int(mine)}-{int(theirs)}"
    return {
        'game_id': str(game['Game Id']),
        'week': int(game['Week']),
        'opponent': opponent,
        'label': f"W{int(game['Week'])} {prefix}{opponent}",
        'help': f"Open the box score — {game.get('Date Display') or ''} {prefix}{opponent}{score}".strip(),
        'won': won,
    }
