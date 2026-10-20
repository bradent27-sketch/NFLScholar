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
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(',')]
    lo, hi = a.weeks.split('-') if '-' in a.weeks else (a.weeks, a.weeks)
    weeks = list(range(int(lo), int(hi) + 1))
    feats = frozenset(DEFAULT_FEATURES | REPLAY)
    scoring_col = 'fantasy_points_ppr' if a.scoring != 'Standard' else 'fantasy_points'
    board_path = a.out + '.board.csv'
    defense_path = a.out + '.defense.parquet'
    align_path = a.out + '.alignment.parquet'
    store = {'rows': [], 'pending': {}, 'year': None, 'week': None, 'align_rows': []}
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
            print(f'{year} w{week}: {n} stat rows, {len(rows)} total, {time.time() - t0:.0f}s', flush=True)
    print('done', len(rows), flush=True)


if __name__ == '__main__':
    main()
