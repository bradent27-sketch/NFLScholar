"""
Strength of schedule, split by position group and computable over any week
range.

TWO THINGS MAKE THIS DIFFERENT FROM A GENERIC SOS NUMBER, and both matter
enough that a single team-level figure is close to useless for drafting:

1. POSITION GROUP. A defense that is brutal against the run and porous
   against the pass is a bad draw for a running back and a gift for a
   receiver. Collapsing that into one "difficulty" for the whole offense
   averages away the only part that was decision-relevant. So schedules are
   scored separately from the perspective of each position group - backs get
   the rushing schedule, quarterbacks and pass catchers the passing
   schedule, and team defenses the overall one.

2. WEEK RANGE. The season-long number answers a question nobody has. What
   people actually want is either the opening stretch (weeks 1-3, when a
   slow start can bury a team) or the fantasy playoffs (weeks 15-17, which
   is the only stretch that decides anything). Both are just different
   windows on the same calculation, so the range is a parameter rather than
   three hardcoded variants.

Difficulty is measured as fantasy points allowed per game to that position
group, computed from this app's own weekly data for the most recent
completed season. Points allowed rather than yards allowed because points
are what you are actually drafting for, and they already fold in touchdown
rate, volume faced, and game script.
"""
import numpy as np
import pandas as pd
import streamlit as st

# Which schedule a position is graded against. Mirrors how every serious
# tool splits it: the run defense is what matters to a back, the pass
# defense to everyone catching or throwing.
POSITION_SOS_GROUP = {
    'RB': 'rushing',
    'QB': 'passing', 'WR': 'passing', 'TE': 'passing',
    'DST': 'overall',
    # Kickers are famously unpredictable and only loosely coupled to
    # opponent quality, so they get no schedule grade rather than a
    # meaningless one.
    'K': None,
}

# Named windows worth having one click away. Weeks 15-17 is the standard
# fantasy playoff window in a 17-week regular season.
WEEK_PRESETS = {
    'Full season': (1, 17),
    'First 3 weeks': (1, 3),
    'Fantasy playoffs (15-17)': (15, 17),
    'Second half (10-17)': (10, 17),
}


@st.cache_data(show_spinner=False)
def build_defense_difficulty(baseline_season, scoring_ppr=1.0):
    """
    Fantasy points allowed per game by every defense, to each position
    group, from the most recent completed season.

    Returns a frame indexed by defensive team with 'rushing', 'passing' and
    'overall' columns, plus a 0-100 percentile for each where 100 is the
    softest matchup.
    """
    from data.loaders import load_weekly_stats_history
    from data.draft_board import score_stats

    hist = load_weekly_stats_history()
    if hist is None or hist.empty or 'opponent_team' not in hist.columns:
        return pd.DataFrame()

    df = hist[pd.to_numeric(hist['week'], errors='coerce').fillna(0) > 0].copy()
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().isin(['REG', 'REGULAR'])]
    df = df[df['season'] == int(baseline_season)]
    if df.empty or 'position' not in df.columns:
        return pd.DataFrame()

    scoring = {
        'pass_yd': 0.04, 'pass_td': 4.0, 'pass_int': -2.0, 'pass_2pt': 2.0,
        'rush_yd': 0.1, 'rush_td': 6.0, 'rush_att': 0.0, 'rush_2pt': 2.0,
        'rec': float(scoring_ppr), 'rec_yd': 0.1, 'rec_td': 6.0, 'rec_2pt': 2.0,
        'te_premium': 0.0, 'fumble_lost': -2.0,
        'fg_0_39': 3.0, 'fg_40_49': 4.0, 'fg_50_plus': 5.0, 'pat': 1.0,
        'bonus_mode': 'cumulative',
    }
    df['_pts'] = score_stats(df, scoring)
    df['position'] = df['position'].astype(str).str.upper()

    games = df.groupby('opponent_team', observed=True)['week'].nunique().replace(0, 1)
    out = pd.DataFrame(index=games.index)

    groups = {'rushing': ['RB'], 'passing': ['QB', 'WR', 'TE'], 'overall': ['QB', 'RB', 'WR', 'TE']}
    for label, positions in groups.items():
        subset = df[df['position'].isin(positions)]
        allowed = subset.groupby('opponent_team', observed=True)['_pts'].sum()
        out[label] = (allowed / games).round(2)

    for label in groups:
        if label in out.columns:
            # Higher percentile = softer defense = better matchup to draft into.
            out[f'{label}_pct'] = out[label].rank(pct=True).mul(100).round(0)
    out.index.name = 'defense'
    return out.reset_index()


@st.cache_data(show_spinner=False)
def build_team_sos(target_season, baseline_season, week_start=1, week_end=17, scoring_ppr=1.0):
    """
    Each NFL team's schedule difficulty over a week range, per position
    group.

    Built from the TARGET season's published schedule (a scheduling fact,
    known before a snap is played) crossed with the BASELINE season's
    measured defensive strength. That split is the honest one: we know
    exactly who plays whom in week 16, and we estimate how hard those
    opponents are from the last season actually played.

    Returns a frame with one row per team: average points allowed by its
    opponents in that window, per group, plus a rank where 1 is the easiest
    schedule.
    """
    from data.loaders import load_schedule

    difficulty = build_defense_difficulty(baseline_season, scoring_ppr=scoring_ppr)
    schedule = load_schedule(int(target_season))
    if difficulty.empty or schedule is None or schedule.empty:
        return pd.DataFrame()

    lookup = difficulty.set_index('defense')
    weeks = pd.to_numeric(schedule['week'], errors='coerce')
    window = schedule[(weeks >= int(week_start)) & (weeks <= int(week_end))]
    if window.empty:
        return pd.DataFrame()

    rows = []
    for _, game in window.iterrows():
        rows.append({'team': game['home_team'], 'opponent': game['away_team'], 'home': True})
        rows.append({'team': game['away_team'], 'opponent': game['home_team'], 'home': False})
    matchups = pd.DataFrame(rows)

    records = []
    for team, group in matchups.groupby('team'):
        record = {'Team': team, 'Games': int(len(group))}
        for label in ('rushing', 'passing', 'overall'):
            if label not in lookup.columns:
                continue
            values = group['opponent'].map(lookup[label])
            record[f'{label}'] = round(float(values.mean()), 2) if values.notna().any() else np.nan
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    for label in ('rushing', 'passing', 'overall'):
        if label in out.columns:
            # Rank 1 = easiest schedule (most points allowed by opponents).
            out[f'{label} rank'] = out[label].rank(ascending=False, method='min').astype('Int64')
    return out.sort_values('overall', ascending=False).reset_index(drop=True)


def attach_sos_to_board(board, sos_df):
    """
    Put each player's own positionally-correct schedule rank on his row.

    One column, not three. A receiver's rushing schedule is noise on his row,
    and showing all three invites reading the wrong one - which is worse than
    not showing it, because it looks authoritative either way.

    BOTH SIDES GO THROUGH standardize_team, which is not decoration. The
    consensus rankings write the Rams as "LAR" and the NFL schedule writes
    them as "LA", so an uppercase string comparison silently dropped every
    Rams player - Puka Nacua's row had a blank SOS while everyone around him
    had a number, which reads as missing data rather than as a join bug. The
    same mismatch exists for JAX/JAC and WAS/WSH.
    """
    from data.odds_sources import standardize_team

    if board.empty or sos_df is None or sos_df.empty:
        out = board.copy()
        out['SOS'] = np.nan
        return out

    out = board.copy()
    sos_teams = sos_df['Team'].map(standardize_team)
    ranks = {}
    for label in ('rushing', 'passing', 'overall'):
        column = f'{label} rank'
        if column in sos_df.columns:
            ranks[label] = {team: value for team, value
                            in zip(sos_teams, sos_df[column]) if team}

    values = []
    for team, pos in zip(out.get('Team', pd.Series(dtype=str)),
                         out['Pos'].astype(str).str.upper()):
        group = POSITION_SOS_GROUP.get(pos)
        if group is None or group not in ranks:
            values.append(np.nan)
            continue
        values.append(ranks[group].get(standardize_team(team), np.nan))
    out['SOS'] = pd.to_numeric(pd.Series(values, index=out.index), errors='coerce')
    return out


def adp_quartiles(board):
    """
    p25 / p75 draft position around each player's ADP.

    The market sites that do this well publish a median and quartiles rather
    than a mean, because draft position is a skewed distribution with a hard
    floor at pick 1 and a long tail - one manager reaching four rounds early
    drags a mean noticeably while barely moving a median. Where a source
    gives us only a mean and a standard deviation, the quartiles are derived
    from it (+/- 0.674 sigma), which is the right conversion for the roughly
    bell-shaped middle of the distribution even though it can't recover the
    tail asymmetry.
    """
    out = board.copy()
    if 'ADP' not in out.columns:
        out['ADP p25'] = np.nan
        out['ADP p75'] = np.nan
        return out
    adp = pd.to_numeric(out['ADP'], errors='coerce')
    sd = pd.to_numeric(out.get('ADP SD'), errors='coerce')
    sd = sd.where(sd.notna() & (sd > 0), np.maximum(2.5, 0.22 * adp))
    out['ADP p25'] = (adp - 0.674 * sd).clip(lower=1).round(1)
    out['ADP p75'] = (adp + 0.674 * sd).round(1)
    return out
