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

from data.utils import calculate_percentile, clean_name_exact, get_val

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
        'opponent': player_rows['opponent_team'].astype(str) if 'opponent_team' in player_rows.columns else '',
    }).groupby('week', as_index=False).agg({
        'targets': 'sum', 'carries': 'sum', 'receptions': 'sum', 'opponent': 'first',
    })
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


def route_efficiency_splits(receiving_summary, receiving_scheme, player_name, route_concept=None, position=None):
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
    export (25+ routes) AT THE SAME POSITION (`position`, when given) -
    a WR's YPRR percentile shouldn't be diluted by every pass-catching RB
    and blocking-first TE in the same file (see _pos_pool). "68th percentile
    YPRR vs man" means among real route-runners at his own position, not
    among everyone with a single snap at any position.

    `route_concept` (PFF's receiving_concept export, `pff['route_concept']`)
    adds Slot YPRR - a THIRD, independent file from the two above, which is
    also where ui.player_snapshot's own "Slot YPRR" tile reads it from (same
    column, same percentile pool - one number, not two implementations of
    it), plus a CALCULATED Wide YPRR (wide_yprr_entry below) - PFF doesn't
    publish Wide as its own route concept, so it's derived as "the rest of
    his routes" once the real slot routes/yards are removed from the season
    total. Slot YPRR renders as a sub-row directly under Slot rate, and Wide
    YPRR as a sub-row under Wide rate - the same "smaller stat living WITH
    its parent row" convention Positional Vulnerability's YPT sub-bar uses,
    per explicit request that these read as companions to the rate above
    them rather than peers of their own.
    """
    out = {'available': False, 'alignment': [], 'scheme': [], 'routes': None, 'yprr': None}
    key = clean_name_exact(pd.Series([player_name])).iloc[0]

    if receiving_summary is not None and not receiving_summary.empty and 'player' in receiving_summary.columns:
        pool = receiving_summary.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        pos_pool = _pos_pool(pool, position) if position else pool
        qualified = pos_pool[_numeric(pos_pool, 'routes').fillna(0) >= 25]
        row = pool[pool['_key'] == key]
        if not row.empty:
            r = row.iloc[0]
            out['available'] = True
            out['routes'] = _float_or_none(r.get('routes'))
            out['yprr'] = _float_or_none(r.get('yprr'))
            slot_rate_entry = _pct_entry('Slot rate', r, 'slot_rate', qualified, '{:.1f}%')
            wide_rate_entry = _pct_entry('Wide rate', r, 'wide_rate', qualified, '{:.1f}%')
            slot_yprr_entry = None
            if route_concept is not None and not route_concept.empty and 'player' in route_concept.columns:
                rc_pool = route_concept.copy()
                rc_pool['_key'] = clean_name_exact(rc_pool['player'])
                rc_pos_pool = _pos_pool(rc_pool, position) if position else rc_pool
                rc_row = rc_pool[rc_pool['_key'] == key]
                if not rc_row.empty:
                    entry = _pct_entry('Slot YPRR', rc_row.iloc[0], 'slot_yprr', rc_pos_pool, '{:.2f}')
                    if entry:
                        entry['sub'] = True
                        slot_yprr_entry = entry
            wide = wide_yprr_entry(receiving_summary, route_concept, player_name, position=position)
            wide_yprr_row = None
            if wide:
                help_text = (
                    "Not published by PFF - calculated as (season yards minus real slot yards) / "
                    "(season routes minus real slot routes)."
                )
                if wide['includes_inline']:
                    help_text += " For a TE this also folds in in-line routes, which PFF has no separate yardage split for."
                wide_yprr_row = {
                    'label': 'Wide YPRR', 'value_str': f"{wide['value']:.2f}", 'pct': wide['pct'],
                    'sub': True, 'help': help_text,
                }
            out['alignment'] = [
                e for e in (
                    slot_rate_entry, slot_yprr_entry, wide_rate_entry, wide_yprr_row,
                    _pct_entry('YPRR', r, 'yprr', qualified, '{:.2f}'),
                    _pct_entry('Yds / Rec', r, 'yards_per_reception', qualified, '{:.1f}'),
                    _pct_entry('ADOT', r, 'avg_depth_of_target', qualified, '{:.1f}'),
                    _pct_entry('Contested catch %', r, 'contested_catch_rate', qualified, '{:.1f}%'),
                ) if e
            ]

    if receiving_scheme is not None and not receiving_scheme.empty and 'player' in receiving_scheme.columns:
        pool = receiving_scheme.copy()
        pool['_key'] = clean_name_exact(pool['player'])
        scheme_pos_pool = _pos_pool(pool, position) if position else pool
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
                man_pool = _numeric(scheme_pos_pool, man_col).dropna()
                zone_pool = _numeric(scheme_pos_pool, zone_col).dropna()
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


# PFF labels running backs "HB" (halfback) in every export this app reads
# (rushing_summary, receiving_summary, offense_blocking - confirmed by
# direct inspection of all three files) - never "RB", which is what this
# app's OWN position column (from nflverse, via stats_df/roster data) uses
# everywhere else. A position filter against a raw PFF export has to
# translate through this alias or it silently matches zero rows and
# produces an empty percentile pool with no error - exactly the class of
# bug HANDOFF.md gotcha #7 warns about ("Silent empty-DataFrame
# fallthrough"). WR/TE need no translation - PFF uses the same letters this
# app does for both.
_PFF_POSITION_ALIAS = {'RB': 'HB'}


def _pff_position(position):
    return _PFF_POSITION_ALIAS.get(str(position).upper(), str(position).upper())


def _pos_pool(df, position, pos_col='position'):
    """
    Filter a PFF export down to one position before it's used as a
    percentile reference pool - every player-side percentile on this tab is
    ranked against same-position players only, never a blended WR/TE/RB
    pool (a TE's blocking grade against a receiving back's route rate isn't
    a fair comparison - they're different jobs). `blocking_grade_entry`
    established the pattern; this is the shared version every other
    player-side percentile pool here goes through too. Returns the frame
    unfiltered if it has no position column to filter on, rather than
    dropping everything.
    """
    if df is None or df.empty or pos_col not in df.columns:
        return df
    return df[df[pos_col].astype(str).str.upper() == _pff_position(position)]


# ---------------------------------------------------------------------------
# Receiver alignment stats PFF doesn't publish - computed here
# ---------------------------------------------------------------------------

def build_wide_yprr_table(receiving_summary, route_concept):
    """
    Per-player "Wide YPRR" - a stat PFF doesn't publish. PFF's route-concept
    export (receiving_concept.csv) breaks out Slot specifically (real,
    measured slot_routes/slot_yards/slot_yprr) but never Wide as its own
    concept, so this is computed as "whatever's left once the real slot
    routes/yards are removed from the season total" - the rest of the
    routes he ran, per explicit request that this be assumed wide.

    That assumption is clean for a WR (in-line snaps are negligible for the
    position) but NOT clean for a TE: a TE's non-slot routes are a mix of
    wide AND in-line alignment, and no export in this app has a route-level
    yardage split for in-line snaps to subtract out separately - so a TE's
    number here is really "wide + in-line combined". `includes_inline`
    flags that per row so a caller's help text can say so, rather than
    silently claiming precision the data doesn't support.

    Vectorized (one merge over the whole pool) rather than a per-player
    lookup loop, since this runs against the full receiving_summary pool
    every time a WR/TE profile is built.
    """
    empty = pd.DataFrame(columns=['player', 'position', 'wide_routes', 'wide_yards', 'wide_yprr', 'includes_inline'])
    if receiving_summary is None or receiving_summary.empty or 'player' not in receiving_summary.columns:
        return empty
    base = receiving_summary[['player', 'position']].copy()
    base['routes'] = _numeric(receiving_summary, 'routes')
    base['yards'] = _numeric(receiving_summary, 'yards')
    if (route_concept is not None and not route_concept.empty
            and {'player', 'slot_routes', 'slot_yards'}.issubset(route_concept.columns)):
        rc = route_concept[['player', 'slot_routes', 'slot_yards']].copy()
        rc['slot_routes'] = _numeric(rc, 'slot_routes').fillna(0)
        rc['slot_yards'] = _numeric(rc, 'slot_yards').fillna(0)
        # Collapsed defensively in case of a duplicate player row (not
        # expected - one row per player in this export) - a plain merge
        # would otherwise silently fan out and double-count.
        rc = rc.groupby('player', as_index=False).sum(numeric_only=True)
        base = base.merge(rc, on='player', how='left')
    else:
        base['slot_routes'] = 0.0
        base['slot_yards'] = 0.0
    base['slot_routes'] = base['slot_routes'].fillna(0)
    base['slot_yards'] = base['slot_yards'].fillna(0)
    base['wide_routes'] = (base['routes'] - base['slot_routes']).clip(lower=0)
    base['wide_yards'] = base['yards'] - base['slot_yards']
    with np.errstate(divide='ignore', invalid='ignore'):
        base['wide_yprr'] = base['wide_yards'] / base['wide_routes'].replace(0, np.nan)
    base['includes_inline'] = base['position'].astype(str).str.upper() == 'TE'
    return base


def wide_yprr_entry(receiving_summary, route_concept, player_name, min_routes=10, position=None):
    """
    One player's calculated Wide YPRR plus its league percentile, ranked
    against every other player AT HIS OWN POSITION (`position`, when given -
    see _pos_pool) with at least `min_routes` estimated wide routes - a
    lower bar than route_efficiency_splits' 25+ routes qualification since
    this is already a subset of a subset for a slot-heavy player. Returns
    None when there's nothing to compute or too thin a pool to rank against
    (see _percentile_of).
    """
    table = build_wide_yprr_table(receiving_summary, route_concept)
    if table.empty:
        return None
    key = clean_name_exact(pd.Series([player_name])).iloc[0]
    table = table.copy()
    table['_key'] = clean_name_exact(table['player'])
    row = table[table['_key'] == key]
    if row.empty:
        return None
    r = row.iloc[0]
    value = _float_or_none(r['wide_yprr'])
    if value is None:
        return None
    pos_table = _pos_pool(table, position) if position else table
    qualified = pos_table[pos_table['wide_routes'] >= min_routes]
    return {
        'value': value,
        'pct': _percentile_of(value, _numeric(qualified, 'wide_yprr').dropna()),
        'routes': float(r['wide_routes']),
        'includes_inline': bool(r['includes_inline']),
    }


def blocking_grade_entry(block_df, player_name, position):
    """
    A skill player's own run-blocking grade, from PFF's offense_blocking
    export (pff['block']) - the same file data.loaders._build_master_pff_grades
    already reads league-wide, just not previously surfaced per-player for
    a skill position. Percentiled within his OWN position only:
    offense_blocking.csv also carries guards/tackles/centers, and ranking a
    receiver's or back's blocking grade against a starting guard's would
    bottom out almost every skill player for no real reason.

    `position` is translated through `_pff_position` before matching - PFF's
    own position column says "HB" for a running back, never "RB".
    """
    if (block_df is None or block_df.empty or 'player' not in block_df.columns
            or 'position' not in block_df.columns or 'grades_run_block' not in block_df.columns):
        return None
    pool = block_df[block_df['position'].astype(str).str.upper() == _pff_position(position)]
    if pool.empty:
        return None
    key = clean_name_exact(pd.Series([player_name])).iloc[0]
    pool = pool.copy()
    pool['_key'] = clean_name_exact(pool['player'])
    row = pool[pool['_key'] == key]
    if row.empty:
        return None
    value = _float_or_none(row.iloc[0].get('grades_run_block'))
    if value is None:
        return None
    return {'value': value, 'pct': _percentile_of(value, _numeric(pool, 'grades_run_block').dropna())}


def receiver_tendency_entries(pff, player_name, position, pct_row):
    """
    Matchup Analyzer's own WR/TE Tendency Profile order - a FORK of
    ui.player_snapshot.build_player_snapshot's WR/TE branch, not a change
    to it: that shared function also feeds Player Search's matrix table and
    Player Compare, both explicitly off-limits (see HANDOFF.md section 8 -
    "Don't change the Player Search tab"), while this ordering/label/
    stat-set request is specific to this tab alone.

    `pct_row` is precompute_league_percentiles' per-player row, already
    filtered to this player+position by the caller - the same source
    Targets/G and Rec/G always read from.

    Order: PFF Rec Grade, Targets/G, Rec/G, RecYd/G, [TE only: Inline%],
    Slot% (+ Slot YPRR / Slot Rec Grade sub-rows), Wide% (+ calculated Wide
    YPRR sub-row), EPA/Target, Success Rate, ADOT, YPRR, Drop Rate, PFF
    Blocking Grade LAST (moved to the bottom per explicit request - it's a
    secondary/situational stat for a skill player, not a headline one, and
    reads oddly sitting second right under his receiving grade). Slot Route
    %/Screen Route %/Screen YPRR from the old shared branch are deliberately
    absent here, per explicit request.

    Every percentile here is ranked within `position` ONLY (`_pos_pool`) -
    receiving_summary/receiving_concept carry WR, TE and pass-catching RBs
    in one file, and ranking a TE's blocking grade or ADOT against a slot
    WR's (or vice versa) isn't a fair comparison; they're different jobs.
    """
    position = str(position).upper()
    entries = []

    def add(label, value_str, pct, sub=False, help_text=None):
        e = {'label': label, 'value_str': value_str, 'pct': pct}
        if sub:
            e['sub'] = True
        if help_text:
            e['help'] = help_text
        entries.append(e)

    rec = pff.get('rec')
    rec_pos_pool = _pos_pool(rec, position) if rec is not None else rec
    row = None
    if rec is not None and not rec.empty and 'player' in rec.columns:
        match = rec[rec['player'].str.lower() == str(player_name).lower()]
        if not match.empty:
            row = match.iloc[0]

    def rec_pct(col, ascending=True):
        if row is None:
            return None
        value = _float_or_none(row.get(col))
        pool = _numeric(rec_pos_pool, col).dropna()
        return _percentile_of(value, pool) if ascending else (
            None if (p := _percentile_of(value, pool)) is None else 100.0 - p)

    if row is not None:
        add('PFF Rec Grade', get_val(row, 'grades_pass_route', "{:.1f}"), rec_pct('grades_pass_route'))

    if pct_row is not None and not pct_row.empty:
        pr = pct_row.iloc[0]
        if pd.notna(pr.get('targets_per_g')):
            add('Targets/G', f"{pr['targets_per_g']:.1f}", pr.get('targets_per_g_pct'))
        if pd.notna(pr.get('rec_per_g')):
            add('Rec/G', f"{pr['rec_per_g']:.1f}", pr.get('rec_per_g_pct'))
        if pd.notna(pr.get('receiving_yards_mean')):
            add('RecYd/G', f"{pr['receiving_yards_mean']:.1f}", pr.get('receiving_yards_mean_pct'))

    if position == 'TE' and row is not None:
        add('Inline%', get_val(row, 'inline_rate', "{:.1f}%"), rec_pct('inline_rate'))

    route_concept = pff.get('route_concept')
    rc_pos_pool = _pos_pool(route_concept, position) if route_concept is not None else route_concept
    rc_row = None
    if route_concept is not None and not route_concept.empty and 'player' in route_concept.columns:
        match = route_concept[route_concept['player'].str.lower() == str(player_name).lower()]
        if not match.empty:
            rc_row = match.iloc[0]

    if row is not None:
        add('Slot%', get_val(row, 'slot_rate', "{:.1f}%"), rec_pct('slot_rate'))
    if rc_row is not None:
        if 'slot_yprr' in route_concept.columns:
            add('Slot YPRR', get_val(rc_row, 'slot_yprr', "{:.2f}"),
                _percentile_of(_float_or_none(rc_row.get('slot_yprr')),
                               _numeric(rc_pos_pool, 'slot_yprr').dropna()), sub=True)
        if 'slot_grades_pass_route' in route_concept.columns:
            add('Slot Rec Grade', get_val(rc_row, 'slot_grades_pass_route', "{:.1f}"),
                _percentile_of(_float_or_none(rc_row.get('slot_grades_pass_route')),
                               _numeric(rc_pos_pool, 'slot_grades_pass_route').dropna()), sub=True)

    if row is not None:
        add('Wide%', get_val(row, 'wide_rate', "{:.1f}%"), rec_pct('wide_rate'))
    wide = wide_yprr_entry(rec, route_concept, player_name, position=position)
    if wide:
        wide_help = ("Not published by PFF - calculated as (season yards minus real slot yards) / "
                     "(season routes minus real slot routes).")
        if wide['includes_inline']:
            wide_help += " For a TE this also folds in in-line routes, which PFF has no separate yardage split for."
        add('Wide YPRR', f"{wide['value']:.2f}", wide['pct'], sub=True, help_text=wide_help)

    if pct_row is not None and not pct_row.empty:
        pr = pct_row.iloc[0]
        if pd.notna(pr.get('epa_per_target')):
            add('EPA/Target', f"{pr['epa_per_target']:.2f}", pr.get('epa_per_target_pct'))
        if pd.notna(pr.get('success_rate_rec')):
            add('Success Rate', f"{pr['success_rate_rec'] * 100:.1f}%", pr.get('success_rate_rec_pct'))

    if row is not None:
        add('ADOT', get_val(row, 'avg_depth_of_target', "{:.1f}"), rec_pct('avg_depth_of_target'))
        add('YPRR', get_val(row, 'yprr', "{:.2f}"), rec_pct('yprr'))
        add('Drop Rate', get_val(row, 'drop_rate', "{:.1f}%"), rec_pct('drop_rate', ascending=False))

    # PFF Blocking Grade last - see docstring. Percentiled within his own
    # position already (blocking_grade_entry filters offense_blocking.csv to
    # `position` before ranking, same _pos_pool convention as everything
    # else above).
    block_entry = blocking_grade_entry(pff.get('block'), player_name, position)
    if block_entry:
        add('PFF Blocking Grade', f"{block_entry['value']:.1f}", block_entry['pct'])

    return entries


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
    This defense's games, as the total `stat_col` it gave up to `position`
    in each - the "are they trending soft or did one blowup carry the
    season number" check that a single season average can't answer.

    `last_n=None` returns the WHOLE season in week order rather than a
    trailing window - what the Positional Vulnerability detail chart wants
    (a full-season trend line), vs. the trailing-window use this function
    was built for originally (still the default, unchanged).
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
    grouped = grouped.sort_values('week')
    return (grouped if last_n is None else grouped.tail(last_n)).reset_index(drop=True)


def league_average_allowed(stats_df, position, stat_col):
    """
    The per-game league average of `stat_col` allowed to `position`, across
    every one of the 32 defenses - the reference line for a WEEK-BY-WEEK
    "what this defense has allowed" chart.

    Deliberately NOT that one defense's own season average (what
    `defense_weekly_allowed`'s caller used to draw as the dashed line): the
    question a week-by-week chart answers is "is this defense soft", and a
    defense's own average can only ever tell you whether one of its games
    was a spike relative to ITSELF - never whether the whole team is soft to
    begin with. A defense that allows 30 fantasy points a week, every week,
    draws a perfectly flat line against its own average and would look
    exactly like a stout unit that also allows 30 a week, every week -
    the league average is what actually separates those two.

    Returns None when there's nothing to average against (a position this
    matrix doesn't track, or a season with no weekly rows yet).
    """
    if stats_df.empty or 'opponent_team' not in stats_df.columns:
        return None
    rows = _played_weeks(stats_df[stats_df['position'].astype(str).str.upper() == str(position).upper()])
    if rows.empty or stat_col not in rows.columns:
        return None
    per_team_game = rows.groupby(['opponent_team', 'week'])[stat_col].sum()
    if per_team_game.empty:
        return None
    return float(per_team_game.mean())


def team_defensive_prowess(run_def, cov_summary):
    """
    team -> 0-100 where HIGHER MEANS SOFTER (a weaker overall defense),
    same direction convention as defense_softness above - but built from
    PFF's overall per-player defensive grade (`grades_defense` - one number
    per player-season, not role-specific, which is why it shows up
    identically in both run_defense_summary AND defense_coverage_summary)
    rather than fantasy points allowed to one position.

    This is deliberately a DIFFERENT axis than defense_softness /
    positional_vulnerability: those are both "how many fantasy points does
    this defense give up AT THIS POSITION", which is exactly the number the
    Defensive Tendency Elasticity curve already buckets against. A second
    curve built from the same points-allowed number wouldn't be measuring
    anything new - it would just be positional softness relabeled. PFF's
    grade is a scouted, position-independent read on the defense itself
    (scheme execution, tackling, coverage technique), which is what "is
    this defense just plain GOOD" is actually asking - a defense can be
    excellent overall and still be the softest matchup for one specific
    position (a great front seven with a beat-up secondary), and that
    distinction is the whole point of offering both curves side by side.

    Snap-weighted per team ACROSS both exports (a run-stuffing lineman's
    run-defense snaps and a slot corner's coverage snaps both count, added
    rather than averaged together first) so a package player who barely
    plays isn't weighted like a three-down starter, and a pure edge rusher
    isn't penalized for having zero coverage snaps to his name.
    """
    frames = []
    if (run_def is not None and not run_def.empty
            and {'team_name', 'grades_defense', 'snap_counts_run'}.issubset(run_def.columns)):
        frames.append(run_def[['team_name', 'grades_defense', 'snap_counts_run']]
                      .rename(columns={'snap_counts_run': 'snaps'}))
    if (cov_summary is not None and not cov_summary.empty
            and {'team_name', 'grades_defense', 'snap_counts_coverage'}.issubset(cov_summary.columns)):
        frames.append(cov_summary[['team_name', 'grades_defense', 'snap_counts_coverage']]
                      .rename(columns={'snap_counts_coverage': 'snaps'}))
    if not frames:
        return {}
    pool = pd.concat(frames, ignore_index=True)
    pool['grades_defense'] = _numeric(pool, 'grades_defense')
    pool['snaps'] = _numeric(pool, 'snaps').fillna(0)
    pool = pool[pool['grades_defense'].notna() & (pool['snaps'] > 0)]
    if pool.empty:
        return {}

    weighted = pool.groupby('team_name').apply(
        lambda g: float((g['grades_defense'] * g['snaps']).sum() / g['snaps'].sum()))
    grades = pd.DataFrame({'team_name': weighted.index, 'grade': weighted.to_numpy()})
    # Ascending on the raw grade (higher grade -> higher raw percentile),
    # then inverted - a LOWER grade is a WEAKER defense, which should read
    # as a HIGHER softness percentile. Same "compute ascending, then flip
    # for a higher-is-better raw stat" convention run_defense_profile
    # already uses for its own grade columns.
    raw_pct = calculate_percentile(grades, 'grade', ascending=True)
    softness_pct = 100.0 - raw_pct
    return dict(zip(grades['team_name'].astype(str), softness_pct.astype(float)))


def ypt_allowed_for_team(positional_coverage, team_name):
    """
    {'WR': {'value', 'pct'}, 'TE': {...}, 'RB': {...}} - yards per target
    allowed to each receiver type, from Sharp's positional coverage export,
    percentiled in the same "higher = softer" direction as everything else
    on this tab.

    Split out from coverage_profile() (which bundles this together with two
    other independent sources behind one shared 'available' flag) because
    Positional Vulnerability wants ONLY this piece, keyed by position rather
    than by coverage_profile's fixed 'vs WR'/'vs TE'/'vs RB' label list.
    """
    out = {}
    if positional_coverage is None or positional_coverage.empty or 'team' not in positional_coverage.columns:
        return out
    nickname = str(team_name).split()[-1] if team_name else ''
    row = positional_coverage[positional_coverage['team'].astype(str).str.lower() == nickname.lower()]
    if row.empty:
        return out
    r = row.iloc[0]
    for pos, col in (('WR', 'ypt_allowed_wr'), ('TE', 'ypt_allowed_te'), ('RB', 'ypt_allowed_rb')):
        value = _float_or_none(r.get(col))
        if value is None:
            continue
        out[pos] = {'value': value, 'pct': _percentile_of(value, _numeric(positional_coverage, col).dropna())}
    return out


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
                # MOF (middle of field) CLOSED = one deep safety, i.e. a
                # SINGLE-HIGH shell; MOF OPEN = two deep safeties splitting
                # the field, i.e. a TWO-HIGH shell - renamed in the UI to
                # the shell names per explicit request, kept as mof_closed/
                # mof_open here since that's what Sharp's own source
                # columns (middle_closed_rate/middle_open_rate) measure.
                'mof_closed': _float_or_none(r.get('middle_closed_rate')),
                'mof_open': _float_or_none(r.get('middle_open_rate')),
                'man_pct': _percentile_of(_float_or_none(r.get('man_rate')),
                                          _numeric(scheme_rates, 'man_rate').dropna()),
                'zone_pct': _percentile_of(_float_or_none(r.get('zone_rate')),
                                           _numeric(scheme_rates, 'zone_rate').dropna()),
                'mof_closed_pct': _percentile_of(_float_or_none(r.get('middle_closed_rate')),
                                                 _numeric(scheme_rates, 'middle_closed_rate').dropna()),
                'mof_open_pct': _percentile_of(_float_or_none(r.get('middle_open_rate')),
                                               _numeric(scheme_rates, 'middle_open_rate').dropna()),
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


def man_zone_grade_rows(scheme, pff_cov):
    """
    The defense's Man vs Zone read laid out in the same
    {'label', 'left', 'right', 'left_str', 'right_str'} row shape
    ui.charts.render_split_bars/the player side's own vs-Man/vs-Zone panel
    uses - Rate (Sharp's scheme dict), then Coverage Grade and QB Rating
    Allowed (PFF, snap-weighted - see _team_coverage_grades) - so the
    defense's man/zone comparison is drawn exactly like the player's,
    per explicit request. Man is always `left`, Zone always `right`.
    """
    rows = []
    man_rate, zone_rate = scheme.get('man_rate'), scheme.get('zone_rate')
    if man_rate is not None or zone_rate is not None:
        rows.append({
            'label': 'Rate', 'left': scheme.get('man_pct'), 'right': scheme.get('zone_pct'),
            'left_str': f"{man_rate:.1f}%" if man_rate is not None else '--',
            'right_str': f"{zone_rate:.1f}%" if zone_rate is not None else '--',
        })
    man_grade, zone_grade = pff_cov.get('man_grade'), pff_cov.get('zone_grade')
    if man_grade or zone_grade:
        rows.append({
            'label': 'Coverage Grade',
            'left': man_grade['pct'] if man_grade else None, 'right': zone_grade['pct'] if zone_grade else None,
            'left_str': f"{man_grade['value']:.1f}" if man_grade else '--',
            'right_str': f"{zone_grade['value']:.1f}" if zone_grade else '--',
        })
    man_rtg, zone_rtg = pff_cov.get('man_rating_allowed'), pff_cov.get('zone_rating_allowed')
    if man_rtg or zone_rtg:
        rows.append({
            'label': 'QB Rating Allowed',
            'left': man_rtg['pct'] if man_rtg else None, 'right': zone_rtg['pct'] if zone_rtg else None,
            'left_str': f"{man_rtg['value']:.1f}" if man_rtg else '--',
            'right_str': f"{zone_rtg['value']:.1f}" if zone_rtg else '--',
        })
    return rows


def defense_alignment_allowed(defense_coverage_summary, slot_coverage, defense_team_pff):
    """
    Real targets/receptions/yards/TDs allowed, split Slot vs the rest of
    the field ("Wide") - replaces a single yards-per-target rate (which
    carries no sense of volume - the actual complaint that motivated this)
    with the fuller allowed line.

    PFF's slot_coverage export (pff['slot_cov']) is real, measured
    slot-alignment defense. PFF has no equivalent outside/wide export, so
    that side is computed the same way this app's Wide YPRR receiver-side
    stat is: the defense's TOTAL coverage numbers (defense_coverage_summary,
    every alignment, pff['cov_summary']) minus the real slot number. Same
    caveat as that stat's docstring applies here too - a safety playing
    middle-of-field zone isn't "outside" alignment either, so this is the
    closest measured equivalent this app has, not a literal boundary-only
    stat.

    `defense_team_pff` must already be a PFF team code (abbr_to_pff_team),
    same convention coverage_profile's own pff_coverage_scheme lookup uses.
    """
    empty = {'available': False, 'rows': []}
    cov_cols = ['receptions', 'targets', 'yards', 'touchdowns']
    if (defense_coverage_summary is None or defense_coverage_summary.empty
            or 'team_name' not in defense_coverage_summary.columns
            or not set(cov_cols).issubset(defense_coverage_summary.columns)):
        return empty

    total_src = defense_coverage_summary.copy()
    for c in cov_cols:
        total_src[c] = _numeric(total_src, c)
    total = total_src.groupby('team_name')[cov_cols].sum()

    if (slot_coverage is not None and not slot_coverage.empty
            and 'team_name' in slot_coverage.columns and set(cov_cols).issubset(slot_coverage.columns)):
        slot_src = slot_coverage.copy()
        for c in cov_cols:
            slot_src[c] = _numeric(slot_src, c)
        slot = slot_src.groupby('team_name')[cov_cols].sum().reindex(total.index, fill_value=0.0)
    else:
        slot = pd.DataFrame(0.0, index=total.index, columns=cov_cols)

    wide = (total - slot).clip(lower=0)
    team = str(defense_team_pff).upper()
    if team not in total.index:
        return empty

    def ypt(frame, idx):
        t = float(frame.loc[idx, 'targets'])
        return float(frame.loc[idx, 'yards']) / t if t > 0 else None

    rows = []
    for col, label in (('targets', 'Targets'), ('receptions', 'Rec'), ('yards', 'Yards'), ('touchdowns', 'TDs')):
        slot_val, wide_val = float(slot.loc[team, col]), float(wide.loc[team, col])
        rows.append({
            'label': label,
            'left': _percentile_of(slot_val, slot[col]), 'right': _percentile_of(wide_val, wide[col]),
            'left_str': f"{slot_val:.0f}", 'right_str': f"{wide_val:.0f}",
        })

    slot_ypt_pool = pd.Series([ypt(slot, t) for t in slot.index if slot.loc[t, 'targets'] > 0], dtype='float64')
    wide_ypt_pool = pd.Series([ypt(wide, t) for t in wide.index if wide.loc[t, 'targets'] > 0], dtype='float64')
    slot_ypt_val, wide_ypt_val = ypt(slot, team), ypt(wide, team)
    rows.append({
        'label': 'Yds / Target',
        'left': _percentile_of(slot_ypt_val, slot_ypt_pool) if slot_ypt_val is not None else None,
        'right': _percentile_of(wide_ypt_val, wide_ypt_pool) if wide_ypt_val is not None else None,
        'left_str': f"{slot_ypt_val:.1f}" if slot_ypt_val is not None else '--',
        'right_str': f"{wide_ypt_val:.1f}" if wide_ypt_val is not None else '--',
    })
    return {'available': True, 'rows': rows}


def defense_stat_rank(stats_df, defense_team, position, stat_col):
    """
    This defense's own per-game average of `stat_col` allowed to
    `position`, plus its league percentile/rank among all 32 defenses - a
    small headline indicator meant to sit above a week-by-week chart so the
    season number has context before the trend line does.

    Works for ANY stat_col, including 'fantasy_points' (which
    ALLOWED_STAT_KEYS doesn't curate an entry for - defense_allowed_by_
    position is restricted to that curated list; this isn't).
    """
    if stats_df.empty or 'opponent_team' not in stats_df.columns:
        return None
    rows = _played_weeks(stats_df[stats_df['position'].astype(str).str.upper() == str(position).upper()])
    if rows.empty or stat_col not in rows.columns:
        return None
    per_team_game = rows.groupby(['opponent_team', 'week'])[stat_col].sum()
    per_team = per_team_game.groupby('opponent_team').mean()
    if str(defense_team) not in per_team.index:
        return None
    value = float(per_team.loc[str(defense_team)])
    return {
        'value': value, 'league_avg': float(per_team.mean()),
        'pct': _percentile_of(value, per_team),
        'rank': int((per_team > value).sum()) + 1, 'of': int(len(per_team)),
    }


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
