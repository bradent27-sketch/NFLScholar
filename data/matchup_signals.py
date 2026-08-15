"""
Compute layer for the Matchup Analyzer tab: everything that decides WHAT to
draw, kept out of ui/ so it can be tested without a Streamlit runtime.

Every function here is pure - it takes already-loaded DataFrames and returns
plain dicts/lists/frames. Nothing in this module reads a file, calls an API,
or touches session state; ui/tabs/matchup_analyzer.py does the loading and
hands the frames in. That's what lets tests/test_matchup_signals.py exercise
the real logic against hand-built fixtures with no network at all.

A note on what this tab is and isn't. It answers "how does THIS player
project against THAT defense", not "who wins this game". Everything shown is
real season data except the two Matchup Curves, which are explicitly labelled
as projections in the UI.

The design is ported from the CFB Scholar Matchup Analyzer, but the data
underneath is not a subset - it's richer in the places that matter most for
NFL prep. The college build has no man/zone receiving splits per player, no
route counts, and no alignment-level yards-per-target allowed. This app has
all three from PFF and Sharp exports that are already on disk, so Route
Efficiency and Coverage Analysis are first-class here rather than the
disclosed approximations they are in the college version.
"""
import numpy as np
import pandas as pd

from data.utils import calculate_percentile, clean_name_exact

# Position -> the stats offered in the player's own game-by-game chart. A
# curated, position-relevant subset, not every column the weekly export
# carries - the chart is for spotting a usage trend at a glance, and a
# 20-entry dropdown defeats that.
GAME_LOG_STATS = {
    'QB': [
        ('passing_yards', 'Pass Yds'), ('passing_tds', 'Pass TDs'),
        ('passing_completions', 'Completions'), ('passing_attempts', 'Attempts'),
        ('passing_interceptions', 'INTs'), ('rushing_yards', 'Rush Yds'),
        ('rushing_tds', 'Rush TDs'), ('fantasy_points', 'Fantasy Pts'),
    ],
    'RB': [
        ('rushing_attempts', 'Carries'), ('rushing_yards', 'Rush Yds'),
        ('rushing_tds', 'Rush TDs'), ('targets', 'Targets'),
        ('receptions', 'Receptions'), ('receiving_yards', 'Rec Yds'),
        ('receiving_tds', 'Rec TDs'), ('fantasy_points', 'Fantasy Pts'),
    ],
    'WR': [
        ('targets', 'Targets'), ('receptions', 'Receptions'),
        ('receiving_yards', 'Rec Yds'), ('receiving_tds', 'Rec TDs'),
        ('rushing_yards', 'Rush Yds'), ('fantasy_points', 'Fantasy Pts'),
    ],
}
GAME_LOG_STATS['TE'] = [e for e in GAME_LOG_STATS['WR'] if e[0] != 'rushing_yards']
GAME_LOG_STATS['FB'] = GAME_LOG_STATS['RB']

# The ONE headline stat per position that the defense's allowed-by-position
# panel and the elasticity curve's opponent axis both key on. Kept separate
# from GAME_LOG_STATS because not every game-log option has a meaningful
# "allowed" counterpart.
MATCHUP_KEY = {
    'QB': ('passing_yards', 'Pass Yds'),
    'RB': ('rushing_yards', 'Rush Yds'),
    'FB': ('rushing_yards', 'Rush Yds'),
    'WR': ('receiving_yards', 'Rec Yds'),
    'TE': ('receiving_yards', 'Rec Yds'),
}

# Per-game stats shown in the defense's "allowed to this position" panel.
ALLOWED_STAT_KEYS = {
    'QB': [('passing_yards', 'Pass Yds'), ('passing_tds', 'Pass TDs'),
           ('passing_completions', 'Comp'), ('rushing_yards', 'Rush Yds')],
    'RB': [('rushing_yards', 'Rush Yds'), ('rushing_tds', 'Rush TDs'),
           ('receptions', 'Rec'), ('receiving_yards', 'Rec Yds')],
    'WR': [('targets', 'Targets'), ('receptions', 'Rec'),
           ('receiving_yards', 'Rec Yds'), ('receiving_tds', 'Rec TDs')],
}
ALLOWED_STAT_KEYS['TE'] = ALLOWED_STAT_KEYS['WR']
ALLOWED_STAT_KEYS['FB'] = ALLOWED_STAT_KEYS['RB']

SUPPORTED_POSITIONS = ('QB', 'RB', 'WR', 'TE', 'FB')


def _numeric(frame, col):
    """A column as floats, or all-NaN of the right length when it's absent -
    lets every aggregation below run without guarding each column, since
    which columns exist genuinely varies by season and by source file."""
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype='float64')
    return pd.to_numeric(frame[col], errors='coerce')


def _played_weeks(frame):
    """Regular-season rows with a real week number, sorted. Weekly exports
    carry postseason rows too, and mixing them into a season average
    silently rewards the four teams that played extra games."""
    if 'week' not in frame.columns:
        return frame.iloc[0:0]
    out = frame.copy()
    out['week'] = pd.to_numeric(out['week'], errors='coerce')
    out = out[out['week'].notna()]
    if 'season_type' in out.columns:
        regular = out['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])
        # Only filter when the column actually distinguishes something -
        # some merged frames carry it as all-NaN, and dropping every row
        # there would blank the whole tab rather than degrade.
        if regular.any():
            out = out[regular]
    return out.sort_values('week')


# ---------------------------------------------------------------------------
# Player side
# ---------------------------------------------------------------------------

def player_game_series(stats_df, name_col, player_name, stat_col):
    """
    One row per played game for this player: week, opponent, and the value
    of `stat_col`.

    A real zero is KEPT, not dropped. A quarterback's 0-TD game is a fact
    about the matchup, and silently skipping it both shortens the bar chart
    and inflates the season average the chart draws its reference line from.
    Rows where the player genuinely has no entry (a bye, an inactive week)
    never existed in the weekly export to begin with, so they're absent for
    the right reason.
    """
    if stats_df.empty or name_col not in stats_df.columns:
        return pd.DataFrame(columns=['week', 'opponent', 'value'])
    rows = stats_df[stats_df[name_col].astype(str) == str(player_name)]
    rows = _played_weeks(rows)
    if rows.empty:
        return pd.DataFrame(columns=['week', 'opponent', 'value'])
    out = pd.DataFrame({
        'week': rows['week'].astype(int),
        'opponent': rows['opponent_team'].astype(str) if 'opponent_team' in rows.columns else '',
        'value': _numeric(rows, stat_col).fillna(0.0),
    })
    # One row per week: a player traded mid-season, or a source with a
    # duplicate row, would otherwise plot the same week twice and shift
    # every later bar's opponent label off by one.
    return out.groupby('week', as_index=False).agg(
        {'opponent': 'first', 'value': 'sum'}).sort_values('week').reset_index(drop=True)


def highlight_games(values, quantile=0.75):
    """
    Which games to star in the game-log chart: this player's own top
    quartile for this stat, not a league threshold. The question the chart
    answers is "when does he go off", which is relative to his own baseline
    - a 70-yard game is a spike for one receiver and a floor for another.

    Returns all-False when every value is identical (including all-zero),
    where "top quartile" would otherwise star the entire season.
    """
    vals = [float(v) for v in values]
    if len(vals) < 4 or len(set(vals)) <= 1:
        return [False] * len(vals)
    cutoff = float(np.quantile(vals, quantile))
    if cutoff <= 0:
        return [False] * len(vals)
    return [v >= cutoff and v > 0 for v in vals]


def usage_and_role(stats_df, name_col, player_name, team, team_col='team'):
    """
    Per-game share of the team's own opportunities: targets, carries, and
    the two combined ("opportunity share"), plus a role-change flag.

    Shares are computed per WEEK against that week's real team totals and
    then averaged, not as a season total over a season total. The two differ
    whenever a player misses games - a receiver who played 8 of 17 weeks
    would otherwise show roughly half his true share, because the
    denominator keeps counting weeks he wasn't on the field.

    `role_change` compares the last three played games' opportunity share
    against everything before them, and only fires on a swing of 8+ points
    off a base of at least 3 games. Below that it's noise: one injury week
    or one blowout moves a three-game window several points on its own.
    """
    empty = {'available': False, 'reason': 'No weekly data for this player yet.'}
    if stats_df.empty or name_col not in stats_df.columns or team_col not in stats_df.columns:
        return empty
    team_rows = _played_weeks(stats_df[stats_df[team_col].astype(str).str.upper() == str(team).upper()])
    player_rows = team_rows[team_rows[name_col].astype(str) == str(player_name)]
    if player_rows.empty:
        return empty

    team_totals = team_rows.groupby('week').agg(
        team_targets=('targets', 'sum'), team_carries=('rushing_attempts', 'sum'),
    ) if {'targets', 'rushing_attempts'}.issubset(team_rows.columns) else pd.DataFrame()
    if team_totals.empty:
        return empty

    per_week = pd.DataFrame({
        'week': player_rows['week'].astype(int),
        'targets': _numeric(player_rows, 'targets').fillna(0.0),
        'carries': _numeric(player_rows, 'rushing_attempts').fillna(0.0),
        'receptions': _numeric(player_rows, 'receptions').fillna(0.0),
    }).groupby('week', as_index=False).sum()
    per_week = per_week.merge(team_totals, left_on='week', right_index=True, how='left')

    def share(num, den):
        d = per_week[den].replace(0, np.nan)
        return (per_week[num] / d * 100).replace([np.inf, -np.inf], np.nan)

    per_week['target_share'] = share('targets', 'team_targets')
    per_week['carry_share'] = share('carries', 'team_carries')
    opp = per_week['targets'] + per_week['carries']
    team_opp = (per_week['team_targets'] + per_week['team_carries']).replace(0, np.nan)
    per_week['opportunity_share'] = (opp / team_opp * 100).replace([np.inf, -np.inf], np.nan)

    catch_rate = (per_week['receptions'].sum() / per_week['targets'].sum() * 100
                  if per_week['targets'].sum() > 0 else None)

    role_change = None
    opp_series = per_week['opportunity_share'].dropna()
    if len(opp_series) >= 6:
        recent, prior = opp_series.iloc[-3:].mean(), opp_series.iloc[:-3].mean()
        delta = recent - prior
        if abs(delta) >= 8:
            role_change = {
                'direction': 'up' if delta > 0 else 'down',
                'delta': float(delta), 'recent': float(recent), 'prior': float(prior),
            }

    return {
        'available': True,
        'games': int(len(per_week)),
        'target_share': _mean_or_none(per_week['target_share']),
        'carry_share': _mean_or_none(per_week['carry_share']),
        'opportunity_share': _mean_or_none(per_week['opportunity_share']),
        'catch_rate': float(catch_rate) if catch_rate is not None else None,
        'role_change': role_change,
        'weekly': per_week,
    }


def _mean_or_none(series):
    clean = series.dropna()
    return float(clean.mean()) if len(clean) else None


def route_efficiency_splits(receiving_summary, receiving_scheme, player_name):
    """
    A receiver's efficiency broken out two independent ways: by ALIGNMENT
    (how often he lines up slot vs wide, from PFF's receiving_summary) and
    by COVERAGE FACED (yards per route run against man vs zone, from PFF's
    receiving_scheme export).

    These are two separate files with no cross-tab between them, so there is
    deliberately no "YPRR from the slot against zone" number here - that
    would require a joint distribution neither export publishes, and
    inventing it by multiplying the two margins would be a fabrication that
    looks like a measurement. The UI shows them as two panels for the same
    reason.

    Percentiles are computed against every qualifying receiver in the same
    export (25+ routes), so "68th percentile YPRR vs man" means among real
    route-runners, not among everyone with a single snap.
    """
    out = {'available': False, 'alignment': [], 'scheme': [], 'routes': None, 'yprr': None}
    key = clean_name_exact(pd.Series([player_name])).iloc[0]

    if receiving_summary is not None and not receiving_summary.empty and 'player' in receiving_summary.columns:
        pool = receiving_summary.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        qualified = pool[_numeric(pool, 'routes').fillna(0) >= 25]
        row = pool[pool['_key'] == key]
        if not row.empty:
            r = row.iloc[0]
            out['available'] = True
            out['routes'] = _float_or_none(r.get('routes'))
            out['yprr'] = _float_or_none(r.get('yprr'))
            out['alignment'] = [
                e for e in (
                    _pct_entry('Slot rate', r, 'slot_rate', qualified, '{:.1f}%'),
                    _pct_entry('Wide rate', r, 'wide_rate', qualified, '{:.1f}%'),
                    _pct_entry('YPRR', r, 'yprr', qualified, '{:.2f}'),
                    _pct_entry('Yds / Rec', r, 'yards_per_reception', qualified, '{:.1f}'),
                    _pct_entry('ADOT', r, 'avg_depth_of_target', qualified, '{:.1f}'),
                    _pct_entry('Contested catch %', r, 'contested_catch_rate', qualified, '{:.1f}%'),
                ) if e
            ]

    if receiving_scheme is not None and not receiving_scheme.empty and 'player' in receiving_scheme.columns:
        pool = receiving_scheme.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        row = pool[pool['_key'] == key]
        if not row.empty:
            r = row.iloc[0]
            out['available'] = True
            for label, man_col, zone_col, fmt in (
                ('YPRR', 'man_yprr', 'zone_yprr', '{:.2f}'),
                ('Yds / Rec', 'man_yards_per_reception', 'zone_yards_per_reception', '{:.1f}'),
                ('Catch %', 'man_caught_percent', 'zone_caught_percent', '{:.1f}%'),
                ('ADOT', 'man_avg_depth_of_target', 'zone_avg_depth_of_target', '{:.1f}'),
                ('Target %', 'man_targets_percent', 'zone_targets_percent', '{:.1f}%'),
            ):
                man_pool = _numeric(pool, man_col).dropna()
                zone_pool = _numeric(pool, zone_col).dropna()
                man_val, zone_val = _float_or_none(r.get(man_col)), _float_or_none(r.get(zone_col))
                if man_val is None and zone_val is None:
                    continue
                out['scheme'].append({
                    'label': label,
                    'left': _percentile_of(man_val, man_pool), 'right': _percentile_of(zone_val, zone_pool),
                    'left_str': fmt.format(man_val) if man_val is not None else '--',
                    'right_str': fmt.format(zone_val) if zone_val is not None else '--',
                })
    return out


def _float_or_none(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(val) else val


def _percentile_of(value, pool):
    """Where `value` sits in `pool`, 0-100. None (not 0) when there's no
    value or too thin a pool to rank against - a percentile invented from
    four samples reads exactly like one computed from four hundred."""
    if value is None or pool is None or len(pool) < 10:
        return None
    return float((pool < value).mean() * 100)


def _pct_entry(label, row, col, pool_df, fmt):
    value = _float_or_none(row.get(col))
    if value is None:
        return None
    return {
        'label': label, 'value_str': fmt.format(value),
        'pct': _percentile_of(value, _numeric(pool_df, col).dropna()),
    }


# ---------------------------------------------------------------------------
# Defense side
# ---------------------------------------------------------------------------

def defense_softness(points_allowed, position):
    """
    team -> 0-100, where HIGHER MEANS SOFTER (more fantasy points allowed to
    this position per game). Same direction as Defensive Yield's own
    strength-of-schedule table and its get_matchup_color scale, so a user
    who has learned "green = good matchup" there reads this identically.

    Returns {} when the matrix has no column for this position - which is
    normal early in a season and for positions the matrix doesn't track.
    """
    if points_allowed is None or points_allowed.empty:
        return {}
    if 'Team' not in points_allowed.columns or position not in points_allowed.columns:
        return {}
    frame = points_allowed[['Team', position]].dropna()
    if frame.empty:
        return {}
    pct = calculate_percentile(frame, position, ascending=True)
    return dict(zip(frame['Team'].astype(str), pct.astype(float)))


def positional_vulnerability(points_allowed, defense_team):
    """
    Which position this defense is worst against - the "who do I actually
    target" read, and the first thing on the defense column for that reason.

    Returns rows sorted softest-first with each position's own per-game
    points allowed, its league percentile, and its plain rank out of the 32
    teams, because a percentile alone doesn't answer "is that 5th-worst or
    just below average".
    """
    if points_allowed is None or points_allowed.empty or 'Team' not in points_allowed.columns:
        return []
    rows = []
    n_teams = points_allowed['Team'].nunique()
    for position in [p for p in ('QB', 'RB', 'WR', 'TE') if p in points_allowed.columns]:
        softness = defense_softness(points_allowed, position)
        if str(defense_team) not in softness:
            continue
        team_row = points_allowed[points_allowed['Team'].astype(str) == str(defense_team)]
        if team_row.empty:
            continue
        value = _float_or_none(team_row.iloc[0][position])
        if value is None:
            continue
        # Rank 1 = allows the MOST, i.e. the softest matchup - stated in the
        # UI caption, since "rank 1 defense" would otherwise read as best.
        rank = int((points_allowed[position] > value).sum()) + 1
        rows.append({
            'position': position, 'pts_allowed': value,
            'pct': softness[str(defense_team)], 'rank': rank, 'of': n_teams,
        })
    return sorted(rows, key=lambda r: r['pct'], reverse=True)


def defense_allowed_by_position(stats_df, defense_team, position):
    """
    Per-game stat line this defense allows to one position, alongside the
    league average for the same position, so every number has a reference
    point rather than being a bare count.

    Games faced is counted as distinct weeks, not summed player rows - a
    defense that faced four different receivers in one game faced ONE game.
    """
    if stats_df.empty or 'opponent_team' not in stats_df.columns or 'position' not in stats_df.columns:
        return {'available': False}
    pos_rows = _played_weeks(stats_df[stats_df['position'].astype(str).str.upper() == str(position).upper()])
    if pos_rows.empty:
        return {'available': False}

    games = pos_rows.groupby('opponent_team')['week'].nunique().replace(0, 1)
    stat_cols = [c for c, _ in ALLOWED_STAT_KEYS.get(str(position).upper(), [])]
    stat_cols = [c for c in stat_cols if c in pos_rows.columns]
    if not stat_cols:
        return {'available': False}
    totals = pos_rows.groupby('opponent_team')[stat_cols].sum()
    per_game = totals.div(games, axis=0)
    if str(defense_team) not in per_game.index:
        return {'available': False}

    entries = []
    for col, label in ALLOWED_STAT_KEYS.get(str(position).upper(), []):
        if col not in per_game.columns:
            continue
        value = float(per_game.loc[str(defense_team), col])
        league = per_game[col].dropna()
        entries.append({
            'label': label, 'value': value, 'league_avg': float(league.mean()),
            'pct': _percentile_of(value, league),
            'rank': int((league > value).sum()) + 1, 'of': int(len(league)),
        })
    return {
        'available': True, 'entries': entries,
        'games': int(games.get(str(defense_team), 0)),
    }


def defense_weekly_allowed(stats_df, defense_team, position, stat_col, last_n=6):
    """
    This defense's most recent games, as the total `stat_col` it gave up to
    `position` in each - the "are they trending soft or did one blowup carry
    the season number" check that a single season average can't answer.
    """
    if stats_df.empty or 'opponent_team' not in stats_df.columns:
        return pd.DataFrame(columns=['week', 'value', 'offense'])
    rows = stats_df[
        (stats_df['opponent_team'].astype(str) == str(defense_team))
        & (stats_df['position'].astype(str).str.upper() == str(position).upper())
    ]
    rows = _played_weeks(rows)
    if rows.empty or stat_col not in rows.columns:
        return pd.DataFrame(columns=['week', 'value', 'offense'])
    grouped = rows.groupby('week').agg(
        value=(stat_col, 'sum'),
        offense=('team', 'first') if 'team' in rows.columns else (stat_col, 'size'),
    ).reset_index()
    return grouped.sort_values('week').tail(last_n).reset_index(drop=True)


def coverage_profile(defense_team, team_name, scheme_rates, positional_coverage, pff_coverage_scheme=None):
    """
    What this defense actually plays, and what it gives up doing it.

    Three independent sources, none of which alone answers the question:
      - Sharp's scheme rates: how often they run man vs zone, and how often
        the middle of the field is open (single-high) vs closed.
      - Sharp's positional coverage: yards per target allowed to WR/TE/RB,
        and separately to outside vs slot alignment.
      - PFF's per-defender coverage-scheme export, aggregated to the team:
        the man and zone grades and QB rating allowed by the players
        actually in coverage.

    All three key teams differently (bare nickname, nickname, and PFF's own
    team codes), which is why this takes both an abbreviation and a full
    name and resolves each source on its own terms.
    """
    out = {'available': False, 'scheme': {}, 'allowed': [], 'alignment': [], 'pff': {}}
    nickname = str(team_name).split()[-1] if team_name else ''

    if scheme_rates is not None and not scheme_rates.empty and 'team' in scheme_rates.columns:
        row = scheme_rates[scheme_rates['team'].astype(str).str.lower() == nickname.lower()]
        if not row.empty:
            r = row.iloc[0]
            out['available'] = True
            out['scheme'] = {
                'man_rate': _float_or_none(r.get('man_rate')),
                'zone_rate': _float_or_none(r.get('zone_rate')),
                'mof_closed': _float_or_none(r.get('middle_closed_rate')),
                'mof_open': _float_or_none(r.get('middle_open_rate')),
                'man_pct': _percentile_of(_float_or_none(r.get('man_rate')),
                                          _numeric(scheme_rates, 'man_rate').dropna()),
            }

    if positional_coverage is not None and not positional_coverage.empty and 'team' in positional_coverage.columns:
        row = positional_coverage[positional_coverage['team'].astype(str).str.lower() == nickname.lower()]
        if not row.empty:
            r = row.iloc[0]
            out['available'] = True
            # A HIGHER yards-per-target allowed is a SOFTER defense, so the
            # percentile is ascending - matching defense_softness' direction
            # so both panels colour the same way.
            for label, col in (('vs WR', 'ypt_allowed_wr'), ('vs TE', 'ypt_allowed_te'), ('vs RB', 'ypt_allowed_rb')):
                value = _float_or_none(r.get(col))
                if value is None:
                    continue
                out['allowed'].append({
                    'label': label, 'value_str': f"{value:.1f}",
                    'pct': _percentile_of(value, _numeric(positional_coverage, col).dropna()),
                })
            outside, slot = _float_or_none(r.get('ypt_allowed_outside')), _float_or_none(r.get('ypt_allowed_slot'))
            if outside is not None or slot is not None:
                out['alignment'] = [{
                    'label': 'Yds / target',
                    'left': _percentile_of(outside, _numeric(positional_coverage, 'ypt_allowed_outside').dropna()),
                    'right': _percentile_of(slot, _numeric(positional_coverage, 'ypt_allowed_slot').dropna()),
                    'left_str': f"{outside:.1f}" if outside is not None else '--',
                    'right_str': f"{slot:.1f}" if slot is not None else '--',
                }]

    if pff_coverage_scheme is not None and not pff_coverage_scheme.empty and 'team_name' in pff_coverage_scheme.columns:
        team_rows = pff_coverage_scheme[pff_coverage_scheme['team_name'].astype(str).str.upper() == str(defense_team).upper()]
        if not team_rows.empty:
            out['available'] = True
            out['pff'] = _team_coverage_grades(team_rows, pff_coverage_scheme)
    return out


def _team_coverage_grades(team_rows, full_pool):
    """
    Team man/zone coverage grade and QB rating allowed, snap-weighted.

    A plain mean over the defenders would let a nickel corner who played 40
    coverage snaps count as much as a starting corner who played 600, which
    is the difference between "this secondary is good" and "one backup had a
    good month". Weighting by each player's own coverage snaps in that
    scheme is the whole point of this aggregation.
    """
    def weighted(grade_col, snap_col):
        grades = _numeric(team_rows, grade_col)
        snaps = _numeric(team_rows, snap_col).fillna(0)
        mask = grades.notna() & (snaps > 0)
        if not mask.any():
            return None
        return float((grades[mask] * snaps[mask]).sum() / snaps[mask].sum())

    def league_pool(grade_col, snap_col):
        values = []
        for _, group in full_pool.groupby('team_name'):
            grades, snaps = _numeric(group, grade_col), _numeric(group, snap_col).fillna(0)
            mask = grades.notna() & (snaps > 0)
            if mask.any():
                values.append((grades[mask] * snaps[mask]).sum() / snaps[mask].sum())
        return pd.Series(values, dtype='float64')

    result = {}
    for key, grade_col, snap_col, ascending in (
        ('man_grade', 'man_grades_coverage_defense', 'man_snap_counts_coverage', True),
        ('zone_grade', 'zone_grades_coverage_defense', 'zone_snap_counts_coverage', True),
        ('man_rating_allowed', 'man_qb_rating_against', 'man_snap_counts_coverage', False),
        ('zone_rating_allowed', 'zone_qb_rating_against', 'zone_snap_counts_coverage', False),
    ):
        value = weighted(grade_col, snap_col)
        if value is None:
            continue
        pool = league_pool(grade_col, snap_col)
        pct = _percentile_of(value, pool)
        # QB rating allowed is inverted: a LOWER number is a better defense,
        # so a high percentile there would colour a great secondary as a
        # soft matchup.
        if pct is not None and not ascending:
            pct = 100 - pct
        result[key] = {'value': value, 'pct': pct}
    return result


def red_zone_defense(pbp_df, defense_team):
    """
    Red-zone trips faced and the touchdown rate allowed on them, from
    play-by-play - the one signal here that no aggregated export carries at
    all, since "how often does a drive inside the 20 finish in the end zone"
    needs field position per play.

    A trip is counted as a distinct (game, drive) pair that reached the
    20 - NOT a count of red-zone plays, which would weight a long grinding
    drive as several trips and a one-play touchdown as one.

    Returns available=False rather than a zero when play-by-play isn't
    reachable (it's a live nflverse pull, unlike everything else on this
    tab), so the UI can say "unavailable" instead of showing a defense that
    allows nothing.
    """
    if pbp_df is None or pbp_df.empty:
        return {'available': False, 'reason': 'Play-by-play data is not available for this season.'}
    needed = {'defteam', 'yardline_100', 'game_id', 'drive'}
    if not needed.issubset(pbp_df.columns):
        return {'available': False, 'reason': 'Play-by-play is missing the drive/field-position columns.'}

    rz = pbp_df[pd.to_numeric(pbp_df['yardline_100'], errors='coerce') <= 20]
    if 'season_type' in rz.columns:
        regular = rz['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])
        if regular.any():
            rz = rz[regular]
    if rz.empty:
        return {'available': False, 'reason': 'No red-zone plays found for this season.'}

    td_col = 'touchdown' if 'touchdown' in rz.columns else None
    per_team = {}
    for team, group in rz.groupby('defteam'):
        trips = group.groupby(['game_id', 'drive'])
        n_trips = trips.ngroups
        if not n_trips:
            continue
        tds = int(trips[td_col].max().fillna(0).sum()) if td_col else 0
        games = group['game_id'].nunique() or 1
        per_team[str(team)] = {
            'trips': n_trips, 'tds': tds, 'td_rate': tds / n_trips * 100,
            'trips_per_game': n_trips / games,
        }
    if str(defense_team) not in per_team:
        return {'available': False, 'reason': 'No red-zone plays found for this defense.'}

    mine = per_team[str(defense_team)]
    td_pool = pd.Series([v['td_rate'] for v in per_team.values()])
    trips_pool = pd.Series([v['trips_per_game'] for v in per_team.values()])
    return {
        'available': True,
        'td_rate': mine['td_rate'], 'td_rate_pct': _percentile_of(mine['td_rate'], td_pool),
        'trips_per_game': mine['trips_per_game'],
        'trips_per_game_pct': _percentile_of(mine['trips_per_game'], trips_pool),
        'trips': mine['trips'], 'tds': mine['tds'],
        'league_td_rate': float(td_pool.mean()),
    }


def run_defense_profile(run_defense_summary, defense_team, sumer_overview=None, team_name=None):
    """
    How this front holds up against the run, from PFF's per-defender run
    defense export aggregated to the team, plus SumerSports' EPA per rush
    allowed where that export covers the season.

    On gap vs zone specifically: this app has NO defense-side gap/zone
    split. PFF publishes gap_attempts/zone_attempts on the RUSHER
    (rushing_summary), never on the defense faced. A gap-vs-zone
    susceptibility number could be manufactured by weighting each defense's
    allowed production by how gap-heavy its opponents happened to be, but
    that's a schedule artefact wearing a scheme label, and it would look
    exactly as authoritative as a measured one. So the defense side reports
    what's real (grade, stop rate, missed tackles, EPA allowed) and the
    player's own gap/zone tendency is shown on the PLAYER side, where it IS
    measured - see scheme_fit().
    """
    out = {'available': False, 'entries': []}
    if run_defense_summary is not None and not run_defense_summary.empty and 'team_name' in run_defense_summary.columns:
        team_rows = run_defense_summary[run_defense_summary['team_name'].astype(str).str.upper() == str(defense_team).upper()]
        if not team_rows.empty:
            out['available'] = True

            def team_weighted(frame, col):
                vals, snaps = _numeric(frame, col), _numeric(frame, 'snap_counts_run').fillna(0)
                mask = vals.notna() & (snaps > 0)
                if not mask.any():
                    return None
                return float((vals[mask] * snaps[mask]).sum() / snaps[mask].sum())

            for label, col, higher_is_better, fmt in (
                ('Run D grade', 'grades_run_defense', True, '{:.1f}'),
                ('Run stop %', 'stop_percent', True, '{:.1f}%'),
                ('Missed tkl %', 'missed_tackle_rate', False, '{:.1f}%'),
                ('Tackle depth', 'avg_depth_of_tackle', False, '{:.1f}'),
            ):
                value = team_weighted(team_rows, col)
                if value is None:
                    continue
                pool = pd.Series([
                    v for v in (team_weighted(g, col) for _, g in run_defense_summary.groupby('team_name'))
                    if v is not None
                ], dtype='float64')
                pct = _percentile_of(value, pool)
                # Percentiles on this panel read as "how soft a matchup" -
                # a strong run defence is a HARD matchup, so a metric where
                # higher means a better defence is inverted here.
                if pct is not None and higher_is_better:
                    pct = 100 - pct
                out['entries'].append({'label': label, 'value_str': fmt.format(value), 'pct': pct})

    if sumer_overview is not None and not sumer_overview.empty and team_name and 'team' in sumer_overview.columns:
        row = sumer_overview[sumer_overview['team'].astype(str).str.lower() == str(team_name).lower()]
        if not row.empty:
            value = _float_or_none(row.iloc[0].get('epa_rush'))
            if value is not None:
                out['available'] = True
                # EPA allowed per rush: higher = the defence is giving more
                # up = softer, which is already the panel's direction.
                out['entries'].append({
                    'label': 'EPA / rush allowed', 'value_str': f"{value:+.2f}",
                    'pct': _percentile_of(value, _numeric(sumer_overview, 'epa_rush').dropna()),
                })
    return out


# ---------------------------------------------------------------------------
# Projections - the only things on this tab that aren't measured facts
# ---------------------------------------------------------------------------

def efficiency_elasticity_curve(series, softness_map, defense_team, label='value'):
    """
    Does this player hold up against good defenses, or does he feast on bad
    ones? Buckets his played games by how soft each opponent was to his
    position, and averages his production in each bucket.

    Three tiers, not a regression: with 17 games there is no fitting a
    meaningful continuous curve, and a straight line through three real
    bucket means is honest about how little is being claimed. The tiers are
    plotted at the MEAN softness of the games actually in them, not at a
    nominal 17/50/83, so a player whose "tough" games were all against
    middling defenses shows that.

    The projection diamond is placed at the selected defense's own softness
    percentile, interpolated between the surrounding tiers - it is a
    read-off of this curve, not a separate model.
    """
    if series is None or series.empty or not softness_map:
        return {'available': False, 'reason': 'Not enough games yet to build a curve.'}
    frame = series.copy()
    frame['softness'] = frame['opponent'].astype(str).map(softness_map)
    frame = frame.dropna(subset=['softness'])
    if len(frame) < 4:
        return {'available': False, 'reason': 'Needs at least 4 games against ranked defenses.'}

    edges = [(0, 33.4, 'Tough'), (33.4, 66.7, 'Average'), (66.7, 100.1, 'Soft')]
    tiers = []
    for lo, hi, name in edges:
        bucket = frame[(frame['softness'] >= lo) & (frame['softness'] < hi)]
        if bucket.empty:
            continue
        tiers.append({
            'x': float(bucket['softness'].mean()), 'y': float(bucket['value'].mean()),
            'n': int(len(bucket)), 'name': name,
        })
    if len(tiers) < 2:
        return {'available': False, 'reason': 'His games are all against similar defenses - no spread to plot.'}

    projection = None
    target = softness_map.get(str(defense_team))
    if target is not None:
        xs = [t['x'] for t in tiers]
        ys = [t['y'] for t in tiers]
        projection = {'x': float(target), 'y': float(np.interp(target, xs, ys))}
    return {
        'available': True, 'tiers': tiers, 'season_avg': float(frame['value'].mean()),
        'projection': projection, 'label': label, 'games': int(len(frame)),
    }


def game_script_sensitivity_curve(series, schedule_df, team, label='value'):
    """
    Whether this player's production depends on the game staying close -
    the read that separates a back who disappears when his team trails from
    one who catches passes in garbage time.

    Games are bucketed by FINAL MARGIN from the subject team's point of
    view (a loss is a negative margin), because the direction matters: "blew
    someone out" and "got blown out" are opposite game scripts and averaging
    them into one "blowout" bucket cancels the signal this is looking for.
    """
    if series is None or series.empty or schedule_df is None or schedule_df.empty:
        return {'available': False, 'reason': 'No schedule data for this season.'}
    margins = team_game_margins(schedule_df, team)
    if not margins:
        return {'available': False, 'reason': 'No completed games with scores yet.'}

    frame = series.copy()
    frame['margin'] = frame['week'].map(margins)
    frame = frame.dropna(subset=['margin'])
    if len(frame) < 4:
        return {'available': False, 'reason': 'Needs at least 4 completed games.'}

    buckets = [
        (-999, -7.5, 'Trailed big', 12.5),
        (-7.5, 0, 'Lost close', 37.5),
        (0, 7.5, 'Won close', 62.5),
        (7.5, 999, 'Won big', 87.5),
    ]
    tiers = []
    for lo, hi, name, x in buckets:
        bucket = frame[(frame['margin'] > lo) & (frame['margin'] <= hi)]
        if bucket.empty:
            continue
        tiers.append({'x': x, 'y': float(bucket['value'].mean()), 'n': int(len(bucket)), 'name': name})
    if len(tiers) < 2:
        return {'available': False, 'reason': 'Every game so far had a similar script.'}
    return {
        'available': True, 'tiers': tiers, 'season_avg': float(frame['value'].mean()),
        'x_ticks': [(t['x'], t['name']) for t in tiers], 'label': label,
    }


def team_game_margins(schedule_df, team):
    """week -> final margin from `team`'s point of view (positive = won by
    that much). Only completed games; a scheduled-but-unplayed row has no
    margin and must not become a zero, which would read as a tie."""
    if schedule_df is None or schedule_df.empty:
        return {}
    needed = {'week', 'home_team', 'away_team', 'home_score', 'away_score'}
    if not needed.issubset(schedule_df.columns):
        return {}
    team = str(team).upper()
    margins = {}
    for _, row in schedule_df.iterrows():
        home, away = str(row['home_team']).upper(), str(row['away_team']).upper()
        if team not in (home, away):
            continue
        home_score, away_score = _float_or_none(row['home_score']), _float_or_none(row['away_score'])
        if home_score is None or away_score is None:
            continue
        week = _float_or_none(row['week'])
        if week is None:
            continue
        margins[int(week)] = home_score - away_score if team == home else away_score - home_score
    return margins


def anytime_td_projection(series_tds, softness_pct=None, market_prob=None):
    """
    P(scores at least one touchdown), as a Poisson draw on his own per-game
    touchdown rate, nudged by how soft the opponent is to his position.

    Poisson rather than a plain "games with a TD" hit rate for two reasons,
    neither of which is that it produces a bigger number - on a typical line
    it produces a slightly smaller one (a 0.5/game rate gives 39.3%, where 6
    TDs across 12 games with one two-TD game gives an empirical 5/12 =
    41.7%):

      1. It uses the whole count. A two-touchdown game is evidence about
         scoring ability that a binary did-he-score rate throws away.
      2. A rate can be scaled by a matchup adjustment; a hit rate cannot.
         Multiplying "41.7% of games" by 1.2 is not a probability of
         anything, whereas scaling an expected-touchdowns rate is exactly
         what the adjustment below means.

    It's also far steadier early in a season, where a hit rate can only move
    in 1/n steps.

    The opponent adjustment is capped at +/-25%
    - the softness percentile is a season-long fantasy-points signal, not a
    touchdown-specific one, and letting it swing the rate further than that
    would over-claim what it measures.

    `market_prob` (from a real book's anytime-TD price, vig included) is
    passed straight through for side-by-side display, never blended in.
    """
    if series_tds is None or series_tds.empty:
        return {'available': False}
    games = len(series_tds)
    total = float(series_tds['value'].sum())
    if games == 0:
        return {'available': False}
    rate = total / games
    adjustment = 1.0
    if softness_pct is not None:
        adjustment = 1.0 + (float(softness_pct) - 50.0) / 50.0 * 0.25
    adjusted = max(0.0, rate * adjustment)
    prob = 1 - float(np.exp(-adjusted))
    return {
        'available': True, 'base_rate': rate, 'adjusted_rate': adjusted,
        'probability': prob, 'games': games, 'total_tds': total,
        'adjustment': adjustment, 'market_prob': market_prob,
    }


def scheme_fit(position, rushing_summary, receiving_summary, player_name, coverage, run_defense):
    """
    A position-specific read on how this player's own style meets what this
    defense does, with the components exposed rather than rolled into one
    opaque score.

    RB: his gap-vs-zone carry split (PFF rushing_summary, measured on him)
    next to the defense's run-defense profile. Deliberately NOT a single
    "fit score" - see run_defense_profile's docstring: there is no
    defense-side gap/zone data in this app, so a combined number would imply
    a comparison that isn't being made.

    WR/TE: his slot/wide split against what this defense gives up to slot vs
    outside receivers, which IS a real like-for-like comparison - both sides
    are measured on alignment. That produces a genuine differential.
    """
    position = str(position).upper()
    key = clean_name_exact(pd.Series([player_name])).iloc[0]

    if position in ('RB', 'FB'):
        if rushing_summary is None or rushing_summary.empty or 'player' not in rushing_summary.columns:
            return {'available': False}
        pool = rushing_summary.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        row = pool[pool['_key'] == key]
        if row.empty:
            return {'available': False}
        r = row.iloc[0]
        gap = _float_or_none(r.get('gap_attempts')) or 0.0
        zone = _float_or_none(r.get('zone_attempts')) or 0.0
        total = gap + zone
        if total <= 0:
            return {'available': False}
        return {
            'available': True, 'kind': 'run',
            'gap_rate': gap / total * 100, 'zone_rate': zone / total * 100,
            'attempts': total,
            'yco_attempt': _float_or_none(r.get('yco_attempt')),
            'elusive_rating': _float_or_none(r.get('elusive_rating')),
            'breakaway_pct': _float_or_none(r.get('breakaway_percent')),
            'defense': run_defense.get('entries', []) if run_defense else [],
        }

    if position in ('WR', 'TE'):
        if receiving_summary is None or receiving_summary.empty or 'player' not in receiving_summary.columns:
            return {'available': False}
        pool = receiving_summary.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        row = pool[pool['_key'] == key]
        if row.empty:
            return {'available': False}
        r = row.iloc[0]
        slot_rate = _float_or_none(r.get('slot_rate'))
        wide_rate = _float_or_none(r.get('wide_rate'))
        alignment = (coverage or {}).get('alignment') or []
        if slot_rate is None or wide_rate is None or not alignment:
            return {'available': False}
        outside_pct, slot_pct = alignment[0].get('left'), alignment[0].get('right')
        if outside_pct is None or slot_pct is None:
            return {'available': False}
        # His own alignment split used as the weight on the defense's two
        # alignment softness percentiles: a defense soft over the middle
        # matters far more to a slot-heavy receiver than to a boundary X.
        weight = slot_rate + wide_rate
        if weight <= 0:
            return {'available': False}
        weighted = (slot_pct * slot_rate + outside_pct * wide_rate) / weight
        return {
            'available': True, 'kind': 'route',
            'slot_rate': slot_rate, 'wide_rate': wide_rate,
            'defense_slot_pct': slot_pct, 'defense_outside_pct': outside_pct,
            'fit_score': float(weighted),
            'neutral_score': float((slot_pct + outside_pct) / 2),
        }

    return {'available': False}
