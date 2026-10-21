"""
Capture, for EVERY projected player and EVERY stat, the in-season blend's inputs from the shipped model's as-of builds, so
the per-stat current-vs-prior weight (STAT_K / STAT_K_BY_POS) can be swept OFFLINE with no further builds.  (Backlog M2,
2026-10-19; the same idea as capture_qb_td_traces.py, which did it for QB passing TDs only.)

For each (year, week) it builds the board exactly as the harness base arm does (DEFAULT_FEATURES plus the historical
injury/reserve replay, apply_injury=False, as-of the target week) and writes one row per (player, stat):

    current_games, current_rate, prior_rate, current_weight, role_confidence, role_change_confidence   the blend's inputs
    blended_rate, pre_vacancy_projection, vacancy_delta, final_projection                                what it produced
    actual (the player's box-score value of the stat that week; NaN when he has no box score)           what happened

    blended = w * current_rate + (1 - w) * prior_rate,  w = games / (games + k_eff)
    -> any other w re-scores the pre-vacancy stat as  pre_vacancy' = pre_vacancy * blended' / blended

The same builds also write a board-level file, `<out>.board.csv`, with exactly the columns of a `calibration_lab.py --mode dump`
file (board points, actual points, `played`, projected and actual stats, the season anchor inputs), so it loads with
`eval_calibration_schemes.load_dump` and always matches the traces row for row (an older dump drifts from the current model).
Both files are rewritten after every week and the run resumes where it stopped.

    python scripts/capture_stat_blend_traces.py --years 2021,2022,2023,2024,2025 --weeks 3-17
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import (DEFAULT_FEATURES, _ALL_PROJECTION_STATS, _played_weeks_before,  # noqa: E402
                                     build_weekly_projections)
from scripts.calibration_lab import CONTEXT_COLS, EXTRA_ACTUALS  # noqa: E402
from scripts.eval_weekly_model import _actual_points  # noqa: E402

REPLAY = frozenset({'v2_historical_injury_replay', 'v2_historical_reserve_replay'})
TRACE_KEYS = ('build_path', 'current_games', 'current_rate', 'prior_rate', 'raw_prior_rate', 'prior2_weight',
              'current_weight', 'blended_rate', 'role_confidence', 'role_change_confidence', 'matchup_multiplier',
              'pace_multiplier', 'script_multiplier', 'availability_multiplier', 'environment_multiplier',
              'participation_multiplier', 'weather_stat_multiplier', 'pre_vacancy_projection', 'vacancy_delta',
              'pass_capacity_delta', 'top_end_delta', 'final_projection', 'prior_source', 'expected_snap_share',
              'alignment_residual_available', 'defense_current_games', 'defense_prior_games', 'role_overlay',
              'defense_estimator')

# Snap counts for seasons without a local export come from an nflverse download that returns an EMPTY table on any failure,
# which silently drops every snap-based step for that season and every later season that reads it as history (2026-10-21: a
# matchup capture lost 2022's snaps mid-run, which also moved its 2023 and 2024 boards). Captures read them through a disk
# cache, retry, and stop rather than capture degraded boards.
SNAP_CACHE_DIR = os.path.join('.sweeps', 'snap_cache')


def install_snap_guard():
    import data.loaders as ld
    orig = ld._pivot_nflreadpy_snap_counts

    def pivot(year):
        path = os.path.join(SNAP_CACHE_DIR, f'snap_counts_{int(year)}.parquet')
        if os.path.exists(path):
            return pd.read_parquet(path)
        for _attempt in range(6):
            out = orig(year)
            if out is not None and not out.empty:
                os.makedirs(SNAP_CACHE_DIR, exist_ok=True)
                out.to_parquet(path, index=False)
                return out
            time.sleep(30)
        raise RuntimeError(f'snap counts for {year} could not be downloaded; stopping instead of capturing degraded boards')

    ld._pivot_nflreadpy_snap_counts = pivot


# --defense-variants (2026-10-20, calibration audit C2-C5): while each board is built exactly as shipped, the defense
# estimator is re-run under other settings and the overall position-channel ratings are recorded, so the defense profile can
# be re-scored offline. Current season: neutral games x per-game decay. Prior season: neutral games x late-season floor.
DEF_CUR_GRID = [(k, d) for k in (4.0, 8.0, 12.0, 20.0) for d in (0.75, 0.85, 0.95)]
DEF_PRIOR_GRID = [(k, f) for k in (4.0, 12.0, 20.0) for f in (0.6, 0.8, 1.0)]


# --alignment-variants (2026-10-20, calibration audit C1): the PFF alignment WR/TE matchup. This season's alignment-defense
# profiles are rebuilt at other shrinkage values, re-blended with last season's at other weights, and each player's unclipped
# multiplier is recorded under every variant, so the shrinkage, the prior blend, the clips and the strength re-score offline.
ALIGN_SHRINKAGE = (1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
ALIGN_PRIOR_W0 = (0.0, 0.2, 0.4)


def install_alignment_variant_hooks(store):
    """Wrap the alignment-defense loader, the prior blend and the per-player multiplier in data.weekly_projections. The
    board is unchanged; per-call records go to ``store['align_rows']``."""
    import data.weekly_projections as wp
    orig_load, orig_blend = wp.load_weekly_alignment_defense_profiles, wp.blend_defense_profiles_toward_prior
    orig_mult = wp.alignment_defense_residual_multiplier

    def load(year, as_of_week, schedule_df, *args, **kw):
        out = orig_load(year, as_of_week, schedule_df, *args, **kw)
        if as_of_week is not None and as_of_week < 19 and 'shrinkage_games' not in kw:
            from data.pff_alignment import aggregate_alignment_defense_profiles
            tg = out.team_games
            store['align_cur'] = {k: (out.profiles if k == 4.0 or tg is None or tg.empty
                                      else aggregate_alignment_defense_profiles(tg, shrinkage_games=k))
                                  for k in ALIGN_SHRINKAGE}
            store['align_variants'] = {(k, 0.2): prof for k, prof in store['align_cur'].items()}
        return out

    def blend(current, prior, split_col, w0, *args, **kw):
        out = orig_blend(current, prior, split_col, w0, *args, **kw)
        if split_col == 'alignment' and store.get('align_cur'):
            store['align_variants'] = {(k, w): orig_blend(prof, prior, split_col, w, *args, **kw)
                                       for k, prof in store['align_cur'].items() for w in ALIGN_PRIOR_W0}
        return out

    def mult(profiles, **kw):
        out = orig_mult(profiles, **kw)
        variants = store.get('align_variants') or {}
        if not variants or not out.get('candidate_available', False):
            return out
        # Only the forward matchup loop (one call per player and stat) gets variants; the same function also serves the
        # per-past-game evidence views, where 18 variants per call would cost hours. That loop's locals name the player.
        caller = sys._getframe(1).f_locals
        if 'preview_columns' not in caller or 'row_index' not in caller or 'cur' not in caller:
            return out
        try:
            cur_frame, row_index = caller['cur'], caller['row_index']
            player = str(cur_frame.at[row_index, caller.get('name_col', 'Player')])
            team = str(cur_frame.at[row_index, 'Team'])
        except Exception:
            return out
        rec = {'year': store['year'], 'week': store['week'], 'player': player, 'pos': str(kw.get('position')), 'team': team,
               'defense': str(kw.get('defense_team')), 'stat': str(kw.get('stat')),
               'shipped': float(out.get('candidate_multiplier', 1.0)), 'shipped_raw': float(out.get('raw_residual', np.nan))}
        # The lookup keeps a ONE-slot index cache keyed on the profile object; alternating 19 frames would rebuild the
        # index (thousands of rows) on every call. Give each variant its own prebuilt index, then restore the shipped one.
        import data.pff_alignment as pa
        cache = pa._ALIGNMENT_PROFILE_INDEX_CACHE
        shipped_cache = (cache['profiles'], cache['index'])
        idx_store = store.setdefault('align_idx', {})
        for (k, w), prof in variants.items():
            if (k, w) not in idx_store or idx_store[(k, w)][0] is not prof:
                cache['profiles'] = None
                idx_store[(k, w)] = (prof, pa._alignment_profile_index(prof))
            cache['profiles'], cache['index'] = prof, idx_store[(k, w)][1]
            alt = orig_mult(prof, **kw)
            rec[f'raw_{k:g}_{w:g}'] = (float(alt.get('raw_residual', np.nan)) if alt.get('candidate_available', False)
                                       else np.nan)
        cache['profiles'], cache['index'] = shipped_cache
        store['align_rows'].append(rec)
        return out

    wp.load_weekly_alignment_defense_profiles = load
    wp.blend_defense_profiles_toward_prior = blend
    wp.alignment_defense_residual_multiplier = mult


# --xtd-variants (2026-10-20, calibration audit C14): the opportunity-based expected-TD blend for rushing/receiving TDs,
# c * own_rate + (1 - c) * xTD with c = events / (events + XTD_TD_EVENT_K) and events = this season's TDs + 0.7 x last
# season's. The inputs and the output are recorded per player, so the event K, the 0.7 and the own-rate blend re-score offline.
def install_xtd_hooks(store):
    import data.weekly_projections as wp
    orig = wp.xtd_blended_rate

    def xtd(stat, pos, own_rate, cur_games, cur_td_total, prior_td_total, player_names, player_teams, *args, **kw):
        out = orig(stat, pos, own_rate, cur_games, cur_td_total, prior_td_total, player_names, player_teams, *args, **kw)
        try:
            own = np.nan_to_num(np.asarray(own_rate, dtype=float), nan=0.0)
            cur_td = np.nan_to_num(np.asarray(cur_td_total, dtype=float), nan=0.0)
            pri_td = np.nan_to_num(np.asarray(prior_td_total, dtype=float), nan=0.0)
            res = np.asarray(out, dtype=float)
            for i in range(len(res)):
                store['xtd_rows'].append({'year': store['year'], 'week': store['week'], 'player': str(player_names[i]),
                                          'pos': str(pos), 'team': str(player_teams[i]), 'stat': stat, 'own': float(own[i]),
                                          'cur_td': float(cur_td[i]), 'pri_td': float(pri_td[i]), 'out': float(res[i])})
        except Exception:
            pass
        return out

    wp.xtd_blended_rate = xtd


# --recency-variants (2026-10-20, calibration audit C9): a player's own within-season rate weights his games by
# RECENCY_DECAY ** (games ago - 1). The weighted rates are re-computed under other decays and recorded per player and stat,
# so the decay re-scores offline (the blend's game count is the raw count, so only the current rate moves).
RECENCY_GRID = (0.70, 0.75, 0.80, 0.90, 0.95, 1.0)


def install_recency_hooks(store):
    import data.weekly_projections as wp
    orig = wp._weighted_player_rates

    def rates(hist_pos, name_col, stats, as_of_week, *args, **kw):
        out = orig(hist_pos, name_col, stats, as_of_week, *args, **kw)
        shipped = out[0]
        if shipped is None or shipped.empty:
            return out
        try:
            pos = str(hist_pos['position'].astype(str).str.upper().mode().iloc[0])
        except Exception:
            return out
        saved, variants = wp.RECENCY_DECAY, {}
        try:
            for decay in RECENCY_GRID:
                wp.RECENCY_DECAY = decay
                variants[decay] = orig(hist_pos, name_col, stats, as_of_week, *args, **kw)[0]
        finally:
            wp.RECENCY_DECAY = saved
        for stat in [c for c in shipped.columns if c != 'weight_sum']:
            frame = pd.DataFrame({'player': shipped.index.astype(str), 'shipped': shipped[stat].to_numpy(float)})
            for decay, v in variants.items():
                frame[f'cur_{decay:g}'] = v[stat].reindex(shipped.index).to_numpy(float)
            frame['year'], frame['week'], frame['pos'], frame['stat'] = store['year'], store['week'], pos, stat
            store['recency_frames'].append(frame)
        return out

    wp._weighted_player_rates = rates


# --matchup-variants (2026-10-20, calibration audit C6 + C7): the role overlay on the matchup (ROLE_MATCHUP_K 10: role
# evidence / (evidence + K) blends a defense's role rating into its overall one) and the defense estimator's blowout handling
# (DEFENSE_BLOWOUT_WEIGHT_DISCOUNT 0.5 at a 28-point DEFENSE_BLOWOUT_MARGIN, RB/QB/TE) and RB offense prior
# (OFFENSE_PRIOR_GAMES 6). Each player's forward matchup multiplier is recorded under every variant; QB passing (overall
# rating only) gets the variant ratings instead.
ROLE_K_GRID = (2.5, 5.0, 20.0, 40.0, 1e9)
EST_VARIANTS = (('disc', 0.0), ('disc', 0.25), ('disc', 0.75), ('disc', 1.0), ('margin', 21.0), ('margin', 35.0),
                ('oprior', 3.0), ('oprior', 12.0))


def install_matchup_variant_hooks(store):
    import data.weekly_projections as wp
    orig_est, orig_blend = wp.build_team_game_quality_adjusted_matchup, wp.blend_defense_prior
    orig_bw, orig_mult = wp._defense_blowout_team_weeks, wp._continuous_role_adjusted_multiplier

    def bw(schedule_df):
        out = orig_bw(schedule_df)
        store['blowout_sources'].append((out, schedule_df))
        return out

    def est(hist_pos, team_col, stats, as_of_week, recency_floor=0.0, **kw):
        out = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor, **kw)
        try:
            pos = str(hist_pos['position'].astype(str).str.upper().mode().iloc[0])
        except Exception:
            return out
        weeks, variants = kw.get('blowout_team_weeks'), {}
        sched = next((sc for res, sc in store['blowout_sources'] if res is weeks), None) if weeks else None
        saved = (wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT, wp.DEFENSE_BLOWOUT_MARGIN)
        try:
            for kind, val in EST_VARIANTS:
                if kind == 'disc' and weeks:
                    wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT = val
                    variants[(kind, val)] = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor, **kw)
                elif kind == 'margin' and sched is not None:
                    wp.DEFENSE_BLOWOUT_MARGIN = val
                    alt = orig_bw(sched)
                    wp.DEFENSE_BLOWOUT_MARGIN = saved[1]
                    variants[(kind, val)] = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor,
                                                     **dict(kw, blowout_team_weeks=alt))
                elif kind == 'oprior' and kw.get('offense_prior_games') is not None:
                    variants[(kind, val)] = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor,
                                                     **dict(kw, offense_prior_games=val))
                wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT, wp.DEFENSE_BLOWOUT_MARGIN = saved
        finally:
            wp.DEFENSE_BLOWOUT_WEIGHT_DISCOUNT, wp.DEFENSE_BLOWOUT_MARGIN = saved
        store['mpending']['prior' if recency_floor and recency_floor > 0 else 'current'] = (pos, out, variants)
        return out

    def blend(current, prior, evidence, prior_games=wp.DEFENSE_PRIOR_GAMES):
        out = orig_blend(current, prior, evidence, prior_games=prior_games)
        cur_pos, _cur_out, cur_vars = store['mpending'].get('current', (None, None, {}))
        pri_pos, _pri_out, pri_vars = store['mpending'].get('prior', (None, None, {}))
        store['mpending'] = {}
        if cur_pos is None or cur_pos != pri_pos or out is None or out.empty:
            return out
        tables = {v: orig_blend(cur_vars.get(v, current), pri_vars.get(v, prior), evidence, prior_games=prior_games)
                  for v in EST_VARIANTS if v in cur_vars or v in pri_vars}
        store['overall_var'][cur_pos] = tables
        if cur_pos == 'QB' and tables:
            for stat in [c for c in out.columns if c in wp.QB_PASSING_MATCHUP_STATS]:
                frame = pd.DataFrame({'defense': out.index.astype(str), 'shipped': out[stat].to_numpy(float)})
                for (kind, val), tb in tables.items():
                    frame[f'ov_{kind}_{val:g}'] = (tb[stat].reindex(out.index).to_numpy(float) if stat in tb.columns
                                                   else np.nan)
                frame['year'], frame['week'], frame['pos'], frame['stat'] = store['year'], store['week'], cur_pos, stat
                store['qb_rating_frames'].append(frame)
        return out

    def mult(overall, role_tables, role_sizes, opponents, role_weights, stat):
        out = orig_mult(overall, role_tables, role_sizes, opponents, role_weights, stat)
        caller = sys._getframe(1).f_locals
        cur_frame, pos = caller.get('cur'), caller.get('pos')
        if cur_frame is None or pos is None or len(cur_frame) != len(out):
            return out
        name_col = caller.get('name_col', 'Player')
        frame = pd.DataFrame({'player': cur_frame[name_col].astype(str).to_numpy(),
                              'team': cur_frame['Team'].astype(str).to_numpy(), 'shipped': np.asarray(out, dtype=float)})
        saved = wp.ROLE_MATCHUP_K
        try:
            for k in ROLE_K_GRID:
                wp.ROLE_MATCHUP_K = k
                frame[f'k_{k:g}'] = orig_mult(overall, role_tables, role_sizes, opponents, role_weights, stat)
        finally:
            wp.ROLE_MATCHUP_K = saved
        for (kind, val), table in store['overall_var'].get(str(pos), {}).items():
            frame[f'ov_{kind}_{val:g}'] = orig_mult(table, role_tables, role_sizes, opponents, role_weights, stat)
        frame['year'], frame['week'], frame['pos'], frame['stat'] = store['year'], store['week'], str(pos), stat
        store['matchup_frames'].append(frame)
        return out

    wp._defense_blowout_team_weeks = bw
    wp.build_team_game_quality_adjusted_matchup = est
    wp.blend_defense_prior = blend
    wp._continuous_role_adjusted_multiplier = mult

# --qb-blend-variants (2026-10-20, calibration audit C13): the QB1 prior-season rate from data/qb_volume_blend.py (team
# dropbacks x the QB's own style and efficiency, each shrunk by its own K). The blend is re-run under other K values and every
# QB1's replaced prior rate is recorded per stat, so the K values re-score offline.
QB_BLEND_VARIANTS = (('QB_STYLE_BLEND_K', 5.0), ('QB_STYLE_BLEND_K', 20.0), ('QB_STYLE_BLEND_K', 40.0),
                     ('QB_VOLUME_BLEND_K', 30.0), ('QB_VOLUME_BLEND_K', 120.0), ('QB_VOLUME_BLEND_K', 240.0),
                     ('QB_PASS_EFFICIENCY_BLEND_K', 50.0), ('QB_PASS_EFFICIENCY_BLEND_K', 200.0),
                     ('QB_PASS_EFFICIENCY_BLEND_K', 400.0),
                     ('QB_RUSH_EFFICIENCY_BLEND_K', 2.5), ('QB_RUSH_EFFICIENCY_BLEND_K', 10.0),
                     ('QB_RUSH_EFFICIENCY_BLEND_K', 20.0),
                     ('QB_RUSH_TD_BLEND_K', 20.0), ('QB_RUSH_TD_BLEND_K', 80.0), ('QB_RUSH_TD_BLEND_K', 160.0),
                     ('QB_VOLUME_STARTER_FLOOR_DROPBACKS_PER_GAME', 15.0),
                     ('QB_VOLUME_STARTER_FLOOR_DROPBACKS_PER_GAME', 25.0))


def install_qb_blend_hooks(store):
    import data.qb_volume_blend as qvb
    import data.weekly_projections as wp
    orig = wp.blend_qb1_volume

    def blend(identity_keys, team_keys, qb1_mask, prior_history, *args, **kw):
        out = orig(identity_keys, team_keys, qb1_mask, prior_history, *args, **kw)
        caller = sys._getframe(1).f_locals
        cur_frame = caller.get('cur')
        rates = out[0]
        if cur_frame is None or not rates or len(cur_frame) != len(qb1_mask):
            return out
        variants = {}
        for name, val in QB_BLEND_VARIANTS:
            if name == 'QB_STYLE_BLEND_K':      # a default argument, bound when the function was defined
                variants[(name, val)] = orig(identity_keys, team_keys, qb1_mask, prior_history, *args, **dict(kw, k=val))[0]
                continue
            saved = getattr(qvb, name)
            try:
                setattr(qvb, name, val)
                variants[(name, val)] = orig(identity_keys, team_keys, qb1_mask, prior_history, *args, **kw)[0]
            finally:
                setattr(qvb, name, saved)
        name_col = caller.get('name_col', 'Player')
        players = cur_frame[name_col].astype(str).to_numpy()
        teams = cur_frame['Team'].astype(str).to_numpy()
        keep = np.asarray(qb1_mask, dtype=bool)
        for stat, arr in rates.items():
            arr = np.asarray(arr, dtype=float)
            m = keep & np.isfinite(arr)
            if not m.any():
                continue
            frame = pd.DataFrame({'player': players[m], 'team': teams[m], 'shipped': arr[m]})
            for (name, val), vr in variants.items():
                frame[f'{name}_{val:g}'] = np.asarray(vr.get(stat, np.full(len(arr), np.nan)), dtype=float)[m]
            frame['year'], frame['week'], frame['pos'], frame['stat'] = store['year'], store['week'], 'QB', stat
            store['qb_blend_frames'].append(frame)
        return out

    wp.blend_qb1_volume = blend

def install_defense_variant_hooks(store):
    """Wrap the defense estimator and the current/prior blend in data.weekly_projections. The board is unchanged (the
    original result is always returned); the variants go to ``store['rows']`` as long records."""
    import data.weekly_projections as wp
    orig_est, orig_blend = wp.build_team_game_quality_adjusted_matchup, wp.blend_defense_prior

    def est(hist_pos, team_col, stats, as_of_week, recency_floor=0.0, **kw):
        out = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor, **kw)
        try:
            pos = str(hist_pos['position'].astype(str).str.upper().mode().iloc[0])
        except Exception:
            return out
        role = 'prior' if recency_floor and recency_floor > 0 else 'current'
        saved = (wp.DEFENSE_PRIOR_GAMES, wp.RECENCY_DECAY)
        variants = {}
        try:
            for k, x in (DEF_PRIOR_GRID if role == 'prior' else DEF_CUR_GRID):
                wp.DEFENSE_PRIOR_GAMES = k
                if role == 'current':
                    wp.RECENCY_DECAY = x
                    variants[(k, x)] = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=recency_floor, **kw)
                else:
                    variants[(k, x)] = orig_est(hist_pos, team_col, stats, as_of_week, recency_floor=x, **kw)
        finally:
            wp.DEFENSE_PRIOR_GAMES, wp.RECENCY_DECAY = saved
        store['pending'][role] = (pos, variants)
        return out

    def blend(current, prior, evidence, prior_games=wp.DEFENSE_PRIOR_GAMES):
        out = orig_blend(current, prior, evidence, prior_games=prior_games)
        cur_pos, cur_vars = store['pending'].get('current', (None, {}))
        pri_pos, pri_vars = store['pending'].get('prior', (None, {}))
        store['pending'] = {}
        if cur_pos is None or cur_pos != pri_pos or out is None or out.empty:
            return out
        n = pd.Series(evidence, dtype=float)
        for team in out.index:
            for stat in out.columns:
                rec = {'year': store['year'], 'week': store['week'], 'pos': cur_pos, 'defense': str(team), 'stat': stat,
                       'shipped': float(out.at[team, stat]), 'n': float(n.get(team, 0.0))}
                for (k, x), frame in cur_vars.items():
                    rec[f'cur_{k:g}_{x:g}'] = (float(frame.at[team, stat]) if (frame is not None and team in frame.index
                                               and stat in frame.columns) else np.nan)
                for (k, x), frame in pri_vars.items():
                    rec[f'pri_{k:g}_{x:g}'] = (float(frame.at[team, stat]) if (frame is not None and team in frame.index
                                               and stat in frame.columns) else np.nan)
                store['rows'].append(rec)
        return out

    wp.build_team_game_quality_adjusted_matchup = est
    wp.blend_defense_prior = blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2021,2022,2023,2024,2025')
    ap.add_argument('--weeks', default='3-17')
    ap.add_argument('--out', default='.sweeps/stat_blend_traces.parquet')
    ap.add_argument('--scoring', default='Full PPR')
    ap.add_argument('--defense-variants', action='store_true', help='also record the defense profile under other settings')
    ap.add_argument('--alignment-variants', action='store_true', help='also record the PFF alignment WR/TE matchup variants')
    ap.add_argument('--xtd-variants', action='store_true', help='also record the expected-TD blend inputs per player')
    ap.add_argument('--matchup-variants', action='store_true', help='also record role-overlay and blowout matchup variants')
    ap.add_argument('--qb-blend-variants', action='store_true', help='also record the QB1 prior-rate blend under other K values')
    ap.add_argument('--recency-variants', action='store_true', help='also record each current rate under other decays')
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    lo, hi = a.weeks.split('-') if '-' in a.weeks else (a.weeks, a.weeks)
    weeks = list(range(int(lo), int(hi) + 1))
    feats = frozenset(DEFAULT_FEATURES | REPLAY)
    scoring_col = 'fantasy_points_ppr' if a.scoring != 'Standard' else 'fantasy_points'
    board_path = a.out + '.board.csv'
    defense_path = a.out + '.defense.parquet'
    align_path = a.out + '.alignment.parquet'
    xtd_path = a.out + '.xtd.parquet'
    recency_path = a.out + '.recency.parquet'
    matchup_path, qb_rating_path = a.out + '.matchup.parquet', a.out + '.qbratings.parquet'
    qb_blend_path = a.out + '.qbblend.parquet'
    install_snap_guard()
    store = {'rows': [], 'pending': {}, 'year': None, 'week': None, 'align_rows': [], 'xtd_rows': [],
             'recency_frames': [], 'matchup_frames': [], 'qb_rating_frames': [], 'mpending': {}, 'overall_var': {},
             'blowout_sources': [], 'qb_blend_frames': []}
    if a.xtd_variants:
        install_xtd_hooks(store)
    if a.recency_variants:
        install_recency_hooks(store)
    if a.matchup_variants:
        install_matchup_variant_hooks(store)
    if a.qb_blend_variants:
        install_qb_blend_hooks(store)
    if a.defense_variants:
        install_defense_variant_hooks(store)
    if a.alignment_variants:
        install_alignment_variant_hooks(store)
    rows, boards, done = [], [], set()
    if os.path.exists(a.out) and os.path.exists(board_path):    # resume
        prev = pd.read_parquet(a.out)
        prev_board = pd.read_csv(board_path, low_memory=False)
        done = set(zip(prev['year'], prev['week'])) & set(zip(prev_board['year'], prev_board['week']))
        rows = [r for r in prev.to_dict('records') if (r['year'], r['week']) in done]
        boards = [prev_board[(prev_board['year'] == y) & (prev_board['week'] == w)] for y, w in sorted(done)]
        if a.defense_variants:
            if os.path.exists(defense_path):
                prev_def = pd.read_parquet(defense_path)
                done &= set(zip(prev_def['year'], prev_def['week']))
                store['rows'] = [r for r in prev_def.to_dict('records') if (r['year'], r['week']) in done]
            else:
                done = set()
            rows = [r for r in rows if (r['year'], r['week']) in done]
            boards = [b for b in boards if len(b) and (int(b['year'].iloc[0]), int(b['week'].iloc[0])) in done]
        print(f'resuming: {len(done)} weeks already captured', flush=True)
    t0 = time.time()
    for year in years:
        stats_df, _team_col, name_col, _ = load_and_merge_data(year, a.scoring)
        stats_df = stats_df.copy()
        stats_df['_wk'] = pd.to_numeric(stats_df['week'], errors='coerce')
        for week in weeks:
            if (year, week) in done:
                continue
            wk = stats_df[stats_df['_wk'] == week]
            if wk.empty:
                continue
            cols = [s for s in _ALL_PROJECTION_STATS if s in wk.columns]
            agg = wk.groupby(name_col, observed=True)[cols].sum()
            a_cols = [c for c in (*_ALL_PROJECTION_STATS, *EXTRA_ACTUALS) if c in wk.columns]
            sa = wk.groupby(name_col, observed=True)[a_cols].sum()
            actual_pts = _actual_points(stats_df, name_col, week, scoring_col)
            build_weekly_projections.clear()
            store['year'], store['week'], store['pending'] = year, week, {}
            store['align_cur'], store['align_variants'], store['align_idx'] = {}, {}, {}
            store['mpending'], store['overall_var'], store['blowout_sources'] = {}, {}, []
            board, meta = build_weekly_projections(year, week, a.scoring, as_of_week=week, apply_injury=False, features=feats)
            build_weekly_projections.clear()
            if board.empty:
                print(f'{year} w{week}: nothing ({meta.get("reason")})', flush=True)
                continue
            keep = [c for c in CONTEXT_COLS if c in board.columns]
            bd = board[keep].rename(columns=CONTEXT_COLS).copy()
            hist = _played_weeks_before(stats_df, week)
            if not hist.empty and 'fantasy_points' in hist.columns:
                g = pd.to_numeric(hist['fantasy_points'], errors='coerce').groupby(hist[name_col]).agg(['mean', 'count'])
                bd['anchor_mean'] = bd['player'].map(g['mean'])
                bd['anchor_games'] = bd['player'].map(g['count']).fillna(0).astype(int)
            else:
                bd['anchor_mean'], bd['anchor_games'] = np.nan, 0
            bd['actual'] = bd['player'].map(actual_pts)
            bd['played'] = bd['actual'].notna()
            for st in _ALL_PROJECTION_STATS:
                bd[f'p_{st}'] = pd.to_numeric(board[st], errors='coerce').to_numpy() if st in board.columns else np.nan
            for st in a_cols:
                bd[f'a_{st}'] = bd['player'].map(sa[st])
            bd['year'], bd['week'] = year, week
            boards.append(bd)
            n = 0
            for _, r in board.iterrows():
                ex = meta['explanations'].get((r['Player'], r['Pos'], r['Team']))
                if not ex:
                    continue
                played = r['Player'] in agg.index
                for stat, tr in ex['stats'].items():
                    row = {'year': year, 'week': week, 'player': r['Player'], 'pos': r['Pos'], 'team': r['Team'],
                           'stat': stat, 'played': played,
                           'actual': (agg.at[r['Player'], stat] if played and stat in agg.columns else None)}
                    for k in TRACE_KEYS:
                        row[k] = tr.get(k)
                    rows.append(row)
                    n += 1
            out = pd.DataFrame(rows)
            for c in out.columns:                               # mixed object columns break parquet; keep the numerics numeric
                if c not in ('player', 'pos', 'team', 'stat', 'build_path', 'prior_source', 'played'):
                    out[c] = pd.to_numeric(out[c], errors='coerce')
            out['prior_source'] = out['prior_source'].astype(str)
            out.to_parquet(a.out, index=False)
            pd.concat(boards, ignore_index=True).to_csv(board_path, index=False)
            if a.defense_variants and store['rows']:
                pd.DataFrame(store['rows']).to_parquet(defense_path, index=False)
            if a.alignment_variants and store['align_rows']:
                pd.DataFrame(store['align_rows']).to_parquet(align_path, index=False)
            if a.xtd_variants and store['xtd_rows']:
                pd.DataFrame(store['xtd_rows']).to_parquet(xtd_path, index=False)
            if a.recency_variants and store['recency_frames']:
                pd.concat(store['recency_frames'], ignore_index=True).to_parquet(recency_path, index=False)
            if a.qb_blend_variants and store['qb_blend_frames']:
                pd.concat(store['qb_blend_frames'], ignore_index=True).to_parquet(qb_blend_path, index=False)
            if a.matchup_variants and store['matchup_frames']:
                pd.concat(store['matchup_frames'], ignore_index=True).to_parquet(matchup_path, index=False)
                if store['qb_rating_frames']:
                    pd.concat(store['qb_rating_frames'], ignore_index=True).to_parquet(qb_rating_path, index=False)
            print(f'{year} w{week}: {n} stat rows, {len(rows)} total, {time.time() - t0:.0f}s', flush=True)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
