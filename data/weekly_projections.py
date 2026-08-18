"""
This app's own weekly fantasy-points projection model - not FantasyPros'
number, and not a bare average of past fantasy points.

WHAT THIS IS FOR. Player Search already has a next-game projection
(`data.transforms.build_player_projection`) for looking up ONE player at a
time. This module projects the WHOLE skill-position pool for one week at
once, for the Weekly Rankings tab, and goes further than that single-player
model in the inputs it blends: current-vs-prior-season usage (weighted
toward the current season as its own sample grows), snap-share/route-share
role confidence, opponent-allowed rates, pace, and a game-script read
against the Vegas-implied spread for the target week - not just a recent-
form/season-average blend times a matchup multiplier.

DELIBERATELY BUILT BY COMPOSITION, NOT FROM SCRATCH. Every external signal
here reuses a primitive this app already has, tested and in production
elsewhere, rather than re-deriving it:
  - opponent-allowed rates: data.transforms.build_stat_allowed_matrix (the
    same matchup engine build_player_projection already uses)
  - pace: data.loaders.load_team_pace (same source)
  - game script: the SAME bucket edges data.matchup_signals.
    game_script_sensitivity_curve uses (Trailed big / Lost close / Won
    close / Won big, split at +/-7.5), computed here in one vectorized pass
    across the whole player pool instead of that function's one-player-at-
    a-time loop (see _vectorized_game_script_multiplier's docstring - this
    is the exact class of mistake data/draft_projections.py's own history
    already flags: "Vectorized; the per-player loop version dominated the
    whole build").
  - injuries: data.draft_sources.fetch_injury_report (same feed the draft
    board already uses)

DELIBERATELY NOT a flat "multiply by Vegas implied team total" adjustment.
data/odds_market.py's own module docstring documents a real backtest
(748 player-seasons, 2023-2025) where scaling a player's SEASON projection
by his team's implied points made it WORSE at every strength tested - "a
player's own usage history already encodes the quality of the offense he
plays in", and scaling on top of that adds variance rather than signal. The
game-script piece here is a different technique answering a different
question: not "is this offense good", but "does the market expect THIS
game to be close or a blowout, and does THIS player's own history show his
role changing when games go that way" - a personalized read-off of the
player's own measured relationship to game margin (the same "interpolate
this player's own curve at the target value" pattern
`data.matchup_signals.efficiency_elasticity_curve` already uses for
opponent softness), not a league-wide flat multiplier. It's applied only to
volume stats, capped at +/-15%, and silently skipped (multiplier 1.0) for
any player without enough game-margin history or without a posted market
line for the target week - see SCRIPT_CLIP and _vectorized_game_script_multiplier.

ONE SHRINKAGE MECHANISM, REUSED FOR EVERYTHING. Rather than a separate
"blend seasons" step and a separate "regress efficiency toward the mean"
step, every per-game rate (targets, carries, receiving yards, TDs, ...)
goes through the SAME weighted blend in _blended_rate:

    w_current = games_this_season / (games_this_season + K[stat])
    blended   = w_current * this_season_rate + (1 - w_current) * prior

`prior` is the player's OWN prior-season rate when he has one, else the
position's current-season average rate (so a rookie with zero history
lands on the position baseline, not on nothing). K is bigger for noisier
stats (touchdowns, interceptions) so those get shrunk harder on a small
sample - the reason a receiver's third career target doesn't get scored
like an every-down role. This is exactly "the current season outweighs the
past as the sample grows": w_current climbs from 0 toward 1 as
games_this_season grows, automatically, no separate schedule to tune.

ROLE CONFIDENCE (snap share + routes run) doesn't get its own multiplier -
it shrinks or widens K itself (see _role_confidence / K_EFFECTIVE_RANGE): a
player whose recent snap share and PFF route rate both say "every-down
role" gets LESS shrinkage (his own small-sample rate is trusted sooner);
a thin, uncertain role gets MORE. That is what "informs the model" means
here - it changes how much weight the model puts on a player's own numbers,
not a bolt-on adjustment layered on top of an unrelated calculation.

NEVER LEAKS THE TARGET WEEK'S OWN RESULT. Every input (usage rates,
opponent-allowed matrix, snap trend, game-script buckets) is computed off
games with week < as_of_week (defaults to the target week itself). This is
what lets the exact same function double as the backtest harness in
docs/weekly_projections_methodology.md - project week N with only what was
known before week N, then compare to what actually happened.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data.transforms import (
    load_and_merge_data, OFFENSE_PROJECTION_STATS, build_stat_allowed_matrix,
    score_projected_stats,
)
from data.loaders import load_team_pace, load_schedule
from data.utils import clean_name_exact

DRAFTABLE_POSITIONS = ('QB', 'RB', 'WR', 'TE')

# Bigger K = more shrinkage toward the prior for a given sample size - see
# the module docstring's _blended_rate explanation. Counting/volume stats
# (targets, carries, yardage) are self-correlated week to week (a player's
# role doesn't reset each Sunday), so a modest K lets his own small sample
# dominate quickly. Scoring stats are lumpy - a 2-TD game is common, a
# 2-TD-per-game RATE is not - so they get pulled toward the prior harder.
STAT_K = {
    'targets': 3, 'receptions': 3, 'receiving_yards': 3,
    'rushing_attempts': 3, 'rushing_yards': 3,
    'passing_attempts': 3, 'passing_completions': 3, 'passing_yards': 3,
    'receiving_tds': 6, 'rushing_tds': 6, 'passing_tds': 5, 'passing_interceptions': 6,
}

# role_confidence in [0, 1] scales K by this range - a confident every-down
# role shrinks K toward the low end (own rate trusted sooner), a thin/
# uncertain role stretches it toward the high end (leans harder on the
# prior). Bounded so role confidence can move K at most 40% either way -
# it's a nudge on top of sample size, not a replacement for it.
K_EFFECTIVE_RANGE = (0.7, 1.3)

MATCHUP_CLIP = (0.75, 1.3)
PACE_CLIP = (0.85, 1.15)
SCRIPT_CLIP = (0.85, 1.15)
# Which raw stats the game-script read applies to - VOLUME only. Touchdowns
# are excluded: too sparse per player-game to bucket reliably by margin
# without the noise swamping the signal, and pass-rate/completion-quality
# stats for QBs are left alone for the same reason plus the passing_yards
# figure already carries most of the same information.
SCRIPT_ELIGIBLE_STATS = {'targets', 'receptions', 'receiving_yards', 'rushing_attempts', 'rushing_yards'}
# Same bucket edges as data.matchup_signals.game_script_sensitivity_curve -
# this is the same measurement, not a different one (see module docstring).
SCRIPT_BUCKETS = [(-999, -7.5, -12.5), (-7.5, 0, -3.75), (0, 7.5, 3.75), (7.5, 999, 12.5)]

INJURY_MULTIPLIER = {'out': 0.0, 'ir': 0.0, 'doubtful': 0.4, 'questionable': 0.85, 'suspended': 0.0}


def _played_weeks_before(stats_df, as_of_week):
    """Regular-season rows strictly before as_of_week - the no-leakage gate
    every CURRENT-season input in this module goes through."""
    if stats_df.empty or 'week' not in stats_df.columns:
        return stats_df.iloc[0:0]
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    return stats_df[(weeks > 0) & (weeks < as_of_week)].copy()


def _all_played_weeks(stats_df):
    """Every real regular-season row, no cutoff - for the PRIOR season side
    of the blend, which is always used in full (there's no leakage risk in
    a season that already ended)."""
    if stats_df.empty or 'week' not in stats_df.columns:
        return stats_df.iloc[0:0]
    weeks = pd.to_numeric(stats_df['week'], errors='coerce')
    return stats_df[weeks > 0].copy()


def _season_totals(stats_df, name_col, team_col, pos, stats):
    """
    Per-player totals + games played for one position's stat list, one
    groupby - not a per-player loop.

    Also carries Team (most recent team that season - a player's row can
    move teams on a trade) and Games, since every caller needs both.
    """
    rows = stats_df[stats_df['position'].astype(str).str.upper() == pos]
    if rows.empty:
        return pd.DataFrame(columns=[name_col, 'Games', team_col] + stats)
    agg = {s: 'sum' for s in stats if s in rows.columns}
    agg['week'] = 'nunique'
    grouped = rows.groupby(name_col, as_index=False).agg(agg).rename(columns={'week': 'Games'})
    last_team = rows.sort_values('week').groupby(name_col)[team_col].last()
    grouped = grouped.merge(last_team.rename('Team'), left_on=name_col, right_index=True, how='left')
    return grouped


def _recent_rate(stats_df, name_col, pos, stats, n=4):
    """
    Per-player trailing-N-games mean for one position's stat list - the
    WITHIN-season half of the current-season rate (see _in_season_rate).
    One groupby.tail(n), not a per-player loop.
    """
    rows = stats_df[stats_df['position'].astype(str).str.upper() == pos]
    if rows.empty:
        return pd.DataFrame(columns=[name_col] + list(stats))
    recent = rows.sort_values('week').groupby(name_col).tail(n)
    return recent.groupby(name_col, as_index=False)[list(stats)].mean()


def _in_season_rate(cur_total, cur_games, recent_avg):
    """
    This season's per-game rate, recency-weighted WITHIN the season -
    60% trailing-4-game average / 40% full-season average, same split
    data.transforms.build_player_projection already uses for its own
    next-game projection (reused rather than invented, so the two models
    agree on how much a hot/cold streak should matter).

    Why this exists as its own step rather than a flat season total/games:
    a straight season-to-date average is exactly what's biased low for
    almost every player still worth projecting by mid-season - the players
    who remain fantasy-relevant in week 10 are disproportionately ones
    whose ROLE GREW as the season went on (an early-season committee back
    who won the job outright by October, a rookie whose snaps climbed every
    week), and averaging that early, smaller-role stretch in with the rest
    at equal weight understates what he's actually doing now. A backtest
    without this step (season-to-date rate only) showed exactly that: a
    consistent several-point-per-player UNDER-projection across every
    position - see docs/weekly_projections_methodology.md.
    """
    season_avg = np.divide(cur_total, np.maximum(cur_games, 1),
                           out=np.zeros_like(cur_total, dtype=float), where=cur_games > 0)
    recent = np.where(np.isnan(recent_avg), season_avg, recent_avg)
    return 0.6 * recent + 0.4 * season_avg


def _blended_rate(cur_rate, cur_games, prior_rate, pos_rate, stat, role_confidence):
    """
    The one shrinkage formula every stat in this module goes through - see
    the module docstring. All arguments are numpy arrays (vectorized over
    the whole player pool for one stat at a time), never scalars in a loop.

    `cur_rate` is this season's recency-weighted rate (_in_season_rate's
    output), not a bare season total/games - see that function's docstring.
    """
    cur_games = np.maximum(cur_games, 0)
    prior = np.where(np.isnan(prior_rate), pos_rate, prior_rate)
    prior = np.where(np.isnan(prior), 0.0, prior)

    base_k = STAT_K.get(stat, 3)
    lo, hi = K_EFFECTIVE_RANGE
    # role_confidence=1 -> multiplier `lo` (less shrinkage, own rate trusted
    # sooner); role_confidence=0 -> multiplier `hi`.
    k_mult = hi - (hi - lo) * np.clip(role_confidence, 0.0, 1.0)
    k_eff = base_k * k_mult

    w_current = cur_games / (cur_games + k_eff)
    return w_current * cur_rate + (1 - w_current) * prior


def _role_confidence(stats_df, name_col, as_of_week, pos, pff_rec):
    """
    0-1 per player: how much of an every-down role his RECENT snaps and (for
    pass-catchers) PFF's season-to-date route rate say he holds. Feeds
    _blended_rate's K, not a separate multiplier - see module docstring.

    Recent snap share is the last up to 3 PLAYED games before as_of_week,
    not the season average - a role change is exactly what a season average
    would smear out.
    """
    hist = _played_weeks_before(stats_df[stats_df['position'].astype(str).str.upper() == pos], as_of_week)
    if hist.empty or 'weekly_snap_pct' not in hist.columns:
        return pd.Series(dtype=float, name='role_confidence')
    recent = (hist.sort_values('week').groupby(name_col).tail(3)
              .groupby(name_col)['weekly_snap_pct'].mean() / 100.0).clip(0, 1)

    if pos in ('WR', 'TE') and pff_rec is not None and not pff_rec.empty and 'player' in pff_rec.columns:
        routes = pff_rec.copy()
        routes['_key'] = clean_name_exact(routes['player'])
        route_rate = routes.set_index('_key')['route_rate']
        route_rate = pd.to_numeric(route_rate, errors='coerce') / 100.0
        keyed_recent = recent.copy()
        keyed_recent.index = clean_name_exact(pd.Series(keyed_recent.index))
        combined = pd.concat([keyed_recent.rename('snap'), route_rate.rename('route')], axis=1)
        blended = combined.mean(axis=1, skipna=True)
        # Reindex back onto real player names via the same exact-key bridge,
        # falling back to snap share alone for anyone with no PFF route row.
        exact_keys = clean_name_exact(pd.Series(recent.index))
        out = pd.Series(blended.reindex(exact_keys.values).to_numpy(), index=recent.index)
        out = out.fillna(recent)
        return out.rename('role_confidence')

    return recent.rename('role_confidence')


def _team_week_margins(schedule_df):
    """[Team, week, margin] for every team-game in the schedule, vectorized
    (home and away rows built at once, not one team at a time) - the whole-
    pool sibling of data.matchup_signals.team_game_margins."""
    needed = {'week', 'home_team', 'away_team', 'home_score', 'away_score'}
    if schedule_df is None or schedule_df.empty or not needed.issubset(schedule_df.columns):
        return pd.DataFrame(columns=['Team', 'week', 'margin'])
    home_score = pd.to_numeric(schedule_df['home_score'], errors='coerce')
    away_score = pd.to_numeric(schedule_df['away_score'], errors='coerce')
    home = pd.DataFrame({'Team': schedule_df['home_team'], 'week': schedule_df['week'],
                         'margin': home_score - away_score})
    away = pd.DataFrame({'Team': schedule_df['away_team'], 'week': schedule_df['week'],
                         'margin': away_score - home_score})
    both = pd.concat([home, away], ignore_index=True).dropna(subset=['margin'])
    both['week'] = pd.to_numeric(both['week'], errors='coerce')
    return both


def _vectorized_game_script_multiplier(stats_df, name_col, team_col, as_of_week, schedule_df,
                                       target_margins, stat):
    """
    Whole-pool version of data.matchup_signals.game_script_sensitivity_curve
    for one stat: bucket every player's PLAYED games by that game's real
    margin (merged in once, not looked up per player), average by
    (player, bucket) in a single groupby, then read off each player's own
    curve at his team's MARKET-IMPLIED margin for the target week.

    Returns a Series of multipliers indexed by player name, 1.0 (neutral)
    for anyone without enough bucketed history or without a target margin.
    """
    hist = _played_weeks_before(stats_df, as_of_week)
    if hist.empty or stat not in hist.columns or target_margins is None or target_margins.empty:
        return pd.Series(dtype=float)
    margins = _team_week_margins(schedule_df)
    if margins.empty:
        return pd.Series(dtype=float)
    merged = hist.merge(margins, left_on=[team_col, 'week'], right_on=['Team', 'week'], how='inner')
    if merged.empty:
        return pd.Series(dtype=float)

    def _bucket(m):
        for lo, hi, mid in SCRIPT_BUCKETS:
            if lo < m <= hi:
                return mid
        return np.nan
    merged['_bucket'] = merged['margin'].map(_bucket)
    merged = merged.dropna(subset=['_bucket'])
    if merged.empty:
        return pd.Series(dtype=float)

    bucket_means = merged.groupby([name_col, '_bucket'])[stat].mean().reset_index()
    game_counts = merged.groupby(name_col)['week'].nunique()
    season_avg = merged.groupby(name_col)[stat].mean()

    out = {}
    for player, group in bucket_means.groupby(name_col):
        if game_counts.get(player, 0) < 4 or len(group) < 2:
            continue
        target = target_margins.get(player)
        if target is None or pd.isna(target):
            continue
        avg = season_avg.get(player)
        if not avg or avg <= 0:
            continue
        xs, ys = group['_bucket'].to_numpy(), group[stat].to_numpy()
        order = np.argsort(xs)
        projected = float(np.interp(target, xs[order], ys[order]))
        out[player] = float(np.clip(projected / avg, *SCRIPT_CLIP))
    return pd.Series(out, dtype=float)


def _target_margins_by_team(year, week):
    """
    Market-implied margin (from THIS team's own point of view - positive
    means favored) for every team's game in `week`, or {} if no lines are
    posted yet for that far out (the normal case beyond the first few
    weeks - see data/odds_market.py's own docstring on posted coverage).
    Never raises; a missing line just means the game-script read sits out
    for that player, same as every other best-effort signal in this app.
    """
    try:
        from data.odds_market import fetch_game_lines, implied_team_points
        games, meta = fetch_game_lines(year)
        if meta.get('error') or games.empty:
            return {}
        week_games = games[pd.to_numeric(games['week'], errors='coerce') == week]
        per_team = implied_team_points(week_games)
        if per_team.empty:
            return {}
        return dict(zip(per_team['team'], per_team['implied_points'] - per_team['implied_allowed']))
    except Exception:
        return {}


def _week_opponents(schedule_df, week):
    """{team: opponent} for one week, from the real schedule - teams absent
    from this dict are on a bye. Vectorized dict build, not a per-team scan."""
    if schedule_df is None or schedule_df.empty:
        return {}
    wk = schedule_df[pd.to_numeric(schedule_df['week'], errors='coerce') == week]
    if wk.empty:
        return {}
    out = dict(zip(wk['home_team'], wk['away_team']))
    out.update(dict(zip(wk['away_team'], wk['home_team'])))
    return out


def _injury_multipliers(year, week):
    """{player: multiplier} from the injury report AS OF `week` - Out/IR/
    Suspended zero the projection out, Doubtful/Questionable discount it.
    Best-effort: an unreachable feed just means no discount is applied,
    same degrade-gracefully convention as every other live source here.

    Passing `week` (not just `year`) matters for anything other than the
    live current week: fetch_injury_report defaults to each player's LATEST
    designation for the whole season, which for a PAST week is that
    player's end-of-season status, not his status that week - a receiver
    who tore an ACL in week 15 would show 'Out' (and get zeroed) on a week
    3 projection otherwise. Harmless for the live/current week (there's no
    future data to exclude either way).
    """
    try:
        from data.draft_sources import fetch_injury_report
        injuries, _err = fetch_injury_report(year, as_of_week=week)
    except Exception:
        return {}
    if injuries is None or injuries.empty:
        return {}
    status = injuries['Injury Status'].astype(str).str.strip().str.lower()
    mult = status.map(INJURY_MULTIPLIER)
    return dict(zip(injuries['Player'], mult.dropna()))


def _load_pff_receiving(year):
    try:
        from data.loaders import load_pff_data_with_fallback
        pff, source_year = load_pff_data_with_fallback(year)
        return pff.get('rec', pd.DataFrame())
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_weekly_projections(year, week, scoring_mode='Full PPR', as_of_week=None, apply_injury=True):
    """
    This app's own projected stat line + fantasy points for every QB/RB/WR/
    TE with usable history, for one week.

    `as_of_week` defaults to `week` (project week N off weeks < N of the
    same season - the real-time use). Passed explicitly and lower than
    `week` by the backtest harness in docs/weekly_projections_methodology.md
    to validate against a week that's already happened without leaking its
    own result into the projection that's supposed to predict it.

    `apply_injury=False` skips the injury-status discount entirely - THE
    BACKTEST NEEDS THIS. fetch_injury_report always returns each player's
    MOST RECENT designation, not a designation as of `week` - fine (correct,
    even) for a live current-week projection, but when validating a past
    week months or years later it's reading each player's LAST reported
    status of that entire season (often 'Out' - end-of-season IR - or a
    routine 'Questionable' from an unrelated later week), applied to every
    week tested regardless of relevance. Measured impact: this alone was
    quietly zeroing out or discounting roughly 1,000 of ~2,000 skill-
    position players in the 2025 backtest, which was the dominant source of
    this model's apparent bias before it was isolated - see
    docs/weekly_projections_methodology.md.

    Returns (DataFrame, meta). Empty DataFrame with meta['reason'] set
    rather than raising when there isn't enough of a current season yet
    (week 1, or a season with no weekly file at all) - same "not enough
    data" convention as build_player_projection.

    KNOWN GAP: WEEK 1 CAN'T BE PROJECTED BY THIS MODEL. The player pool
    itself (who's on which team, in what role) is read off THIS season's
    own games, same as every other input - with zero of those played yet,
    there's no way to tell a returning starter from someone who lost his
    job in camp. This isn't a threshold that could be tuned lower; it's
    structural. Week 1 (and the preseason generally) is exactly where
    FantasyPros' own weekly projection (data.draft_sources.
    fetch_fantasypros_weekly_projections) is the right tool instead - their
    analysts have a season-opener number, this model doesn't.
    """
    if as_of_week is None:
        as_of_week = week
    stats_df, team_col, name_col, _ = load_and_merge_data(year, scoring_mode)
    if stats_df.empty or 'week' not in stats_df.columns:
        return pd.DataFrame(), {'reason': f'No weekly data for {year} yet.'}
    hist = _played_weeks_before(stats_df, as_of_week)
    if hist.empty:
        return pd.DataFrame(), {'reason': f'Week {as_of_week} of {year} has no earlier games this '
                                          'season to project from yet (see this function\'s '
                                          'docstring - week 1 needs FantasyPros\' own weekly '
                                          'projection instead).'}

    prior_stats, prior_team_col, prior_name_col, _ = pd.DataFrame(), team_col, name_col, None
    try:
        prior_stats, prior_team_col, prior_name_col, _ = load_and_merge_data(year - 1, scoring_mode)
    except Exception:
        prior_stats = pd.DataFrame()

    schedule_df = load_schedule(year)
    opponents = _week_opponents(schedule_df, week)
    target_margins = _target_margins_by_team(year, week)
    injury_mult = _injury_multipliers(year, week) if apply_injury else {}
    pff_rec = _load_pff_receiving(year)
    pace = load_team_pace(year)
    league_pace = pace['def_pace'].mean() if pace is not None and not pace.empty and 'def_pace' in pace.columns else None

    all_rows = []
    for pos in DRAFTABLE_POSITIONS:
        stats = OFFENSE_PROJECTION_STATS[pos]
        cur = _season_totals(hist, name_col, team_col, pos, stats)
        if cur.empty:
            continue
        prior = (_season_totals(_all_played_weeks(prior_stats), prior_name_col, prior_team_col, pos, stats)
                 if not prior_stats.empty else pd.DataFrame())
        prior_rates = pd.DataFrame({s: prior[s] / prior['Games'].replace(0, 1) for s in stats}) if not prior.empty else pd.DataFrame()
        if not prior.empty:
            prior_rates[name_col] = prior[name_col].values

        recent = _recent_rate(hist, name_col, pos, stats, n=4).rename(
            columns={s: f'{s}__recent' for s in stats})
        cur = cur.merge(recent, on=name_col, how='left')

        role_conf = _role_confidence(stats_df, name_col, as_of_week, pos, pff_rec)
        cur = cur.merge(role_conf.rename('role_confidence'), left_on=name_col, right_index=True, how='left')
        cur['role_confidence'] = cur['role_confidence'].fillna(0.5)

        allowed_matrix = build_stat_allowed_matrix(hist, position_filter=[pos])
        league_means = allowed_matrix.mean() if not allowed_matrix.empty else pd.Series(dtype=float)

        cur['Opponent'] = cur['Team'].map(opponents)
        cur = cur[cur['Opponent'].notna()].copy()  # bye-week teams drop out entirely
        if cur.empty:
            continue
        cur['target_margin'] = cur['Team'].map(target_margins)

        proj_cols = {}
        for stat in stats:
            if stat not in cur.columns:
                continue
            cur_total = cur[stat].to_numpy(dtype=float)
            cur_games = cur['Games'].to_numpy(dtype=float)
            recent_avg = (cur[f'{stat}__recent'].to_numpy(dtype=float)
                         if f'{stat}__recent' in cur.columns else np.full(len(cur), np.nan))
            in_season_rate = _in_season_rate(cur_total, cur_games, recent_avg)

            if not prior_rates.empty and stat in prior_rates.columns:
                prior_map = pd.Series(prior_rates[stat].to_numpy(), index=prior_rates[name_col])
                prior_rate = cur[name_col].map(prior_map).to_numpy(dtype=float)
            else:
                prior_rate = np.full(len(cur), np.nan)
            # Games-weighted, not a plain mean of each player's own rate - a
            # one-game emergency start would otherwise count exactly as much
            # as a nine-game starter's rate in setting the rookie/no-prior
            # baseline, which understates it (confirmed real in the backtest
            # write-up in docs/weekly_projections_methodology.md).
            games_total = cur_games.sum()
            pos_rate = float(cur_total.sum() / games_total) if games_total > 0 else 0.0
            pos_rate_arr = np.full(len(cur), pos_rate)

            blended = _blended_rate(in_season_rate, cur_games, prior_rate, pos_rate_arr, stat,
                                    cur['role_confidence'].to_numpy(dtype=float))

            matchup_mult = np.ones(len(cur))
            if stat in league_means.index and league_means[stat] > 0:
                opp_allowed = cur['Opponent'].map(allowed_matrix[stat]).fillna(league_means[stat])
                matchup_mult = np.clip(opp_allowed.to_numpy(dtype=float) / league_means[stat], *MATCHUP_CLIP)

            script_mult = np.ones(len(cur))
            if stat in SCRIPT_ELIGIBLE_STATS:
                script_series = _vectorized_game_script_multiplier(
                    stats_df, name_col, team_col, as_of_week, schedule_df, cur.set_index(name_col)['target_margin'], stat)
                if not script_series.empty:
                    script_mult = cur[name_col].map(script_series).fillna(1.0).to_numpy(dtype=float)

            proj_cols[stat] = blended * matchup_mult * script_mult

        pace_mult = pd.Series(1.0, index=cur.index)
        if league_pace and league_pace > 0:
            opp_pace = cur['Opponent'].map(pace['def_pace'])
            pace_mult = np.clip(opp_pace.fillna(league_pace) / league_pace, *PACE_CLIP)

        inj_mult = cur[name_col].map(injury_mult).fillna(1.0)

        for stat in proj_cols:
            proj_cols[stat] = proj_cols[stat] * pace_mult.to_numpy() * inj_mult.to_numpy()

        out = pd.DataFrame({
            'Player': cur[name_col], 'Pos': pos, 'Team': cur['Team'], 'Opponent': cur['Opponent'],
            'Games This Season': cur['Games'].astype(int),
            'Role Confidence': cur['role_confidence'].round(2),
        })
        for stat, values in proj_cols.items():
            out[stat] = np.round(values, 2)
        proj_dicts = out[[s for s in stats if s in out.columns]].to_dict('records')
        out['Model Proj Pts'] = [score_projected_stats(d, scoring_mode) for d in proj_dicts]
        out['Injury Status'] = out['Player'].map(lambda p: 'Out/Doubtful' if injury_mult.get(p, 1.0) < 0.9 else '')
        all_rows.append(out)

    if not all_rows:
        return pd.DataFrame(), {'reason': f'No projectable players found for week {week}.'}
    result = pd.concat(all_rows, ignore_index=True).sort_values('Model Proj Pts', ascending=False).reset_index(drop=True)
    meta = {'reason': None, 'year': year, 'week': week, 'as_of_week': as_of_week,
            'players': int(len(result)), 'scoring': scoring_mode}
    return result, meta
