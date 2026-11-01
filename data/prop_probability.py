"""
The chance a player goes over a posted prop line, from the board's projected stat (2026-10-28, user: "This should have the
capability of building prop specific accuracy for prop betting going forward").

The board's stat columns are MEANS, and their departures from a player's own level overshoot. The deep audit
(scripts/deep_audit.py, .sweeps/deep_audit/REPORT_PASS2.md) measured how much of the gap between the projection and the player's
own average actually came true:
- WR/TE/RB volume: 58-81%;
- QB rushing: about 45%;
- QB passing: about 100%.

The points calibration absorbs this for fantasy points, but a prop is priced on the stat. So a prop probability takes two steps,
both fit by `scripts/deep_audit.py --stage props2` on today's model's boards, 2021-2026:

  1. RECALIBRATE the mean, per position: a ridge fit of the result on the projection, the player's season to date, last 3 games
     and last season. Each history input falls back to the projection where missing, with a missing flag. QB attempts,
     completions and interceptions skip this step: it lost there forward-chained, since QB passing departures come true in full.
  2. Read P(over) from the EMPIRICAL spread of results around that mean at its level: the ratio actual / mean in projection-level
     bins. The Poisson (counts) / Normal (yards) shape that data/market_devig.py assumes is too narrow for volume stats: its 50%
     interval held 35-43% of results for attempts, targets and receptions.

Forward-chained 2023-2026, the recalibration cut CRPS by 0.2-0.9% on targets, receptions, receiving yards, carries, rushing yards
and passing yards.

2026 weeks 3-7, live pre-kickoff boards, engine fit on 2021-2025, closing lines (Pinnacle, then DraftKings):
- Brier 0.255 against the books' de-vigged 0.248. The books' close is the better probability, and the model's P(over) carried
  about a third of the confidence it shows (the best shrink toward 1/2 was 0.34).
- Its side still won 55% on receiving yards and 54% on receptions at a 5%+ gap to the book's price.
- It lost badly on passing yards and attempts (37%).

Read a gap as a lean to track, per stat. scripts/prop_scorecard.py keeps the record week by week.

Engines: data/prop_engines.json (one entry per stat). Refit: run `scripts/deep_audit.py --stage all`, then `--stage props2`, then
`python scripts/prop_scorecard.py --install-engines`.
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from data.utils import clean_name_exact

ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prop_engines.json')
# board stat -> the props archive's market name
PROP_MARKETS = {'passing_yards': 'passing_yards', 'passing_attempts': 'attempts', 'passing_completions': 'passing_completions',
                'passing_tds': 'passing_tds', 'passing_interceptions': 'passing_interceptions',
                'rushing_attempts': 'carries', 'rushing_yards': 'rushing_yards', 'targets': 'targets',
                'receptions': 'receptions', 'receiving_yards': 'receiving_yards'}
# nflverse spellings of a board stat, for a stats frame that lacks the board's own
_ALIASES = {'passing_attempts': ('attempts',), 'passing_completions': ('completions',), 'rushing_attempts': ('carries',),
            'passing_interceptions': ('interceptions',)}
_OFF_POS = ('QB', 'RB', 'WR', 'TE', 'FB')


def load_engines():
    """{stat: engine} from ENGINE_PATH (data/prop_engines.json unless a script points it elsewhere; {} when missing)."""
    return _read_engines(ENGINE_PATH)


@lru_cache(maxsize=4)
def _read_engines(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _game_logs(stats_df, name_col):
    """One row per player-game (regular season, weeks 1-18, offensive positions; FB read as RB, silent games left out: they
    are not box-score games) with the prop stats."""
    s = stats_df.copy()
    if '_silent_game' in s.columns:
        s = s[~s['_silent_game'].fillna(False).astype(bool)]
    s['week'] = pd.to_numeric(s['week'], errors='coerce')
    if 'season_type' in s.columns:
        s = s[s['season_type'].isna() | s['season_type'].astype(str).str.upper().eq('REG')]
    s = s[s['week'].between(1, 18)]
    if 'position' in s.columns:
        s = s[s['position'].isin(_OFF_POS)]
    out = pd.DataFrame({'_key': clean_name_exact(s[name_col].astype(str)).to_numpy(),
                        'pos': s['position'].replace({'FB': 'RB'}).to_numpy() if 'position' in s.columns else None,
                        'week': s['week'].astype(int).to_numpy()})
    for stat in PROP_MARKETS:
        col = stat if stat in s.columns else next((a for a in _ALIASES.get(stat, ()) if a in s.columns), None)
        out[stat] = pd.to_numeric(s[col], errors='coerce').to_numpy() if col else np.nan
    return out.groupby(['_key', 'pos', 'week'])[list(PROP_MARKETS)].sum(min_count=1).reset_index()


def player_history(stats_df, name_col, week, prior_stats_df=None, prior_name_col=None):
    """The history inputs the recalibration reads, per (_key, pos), from box-score games before `week`: h_s2d_<stat> (season
    to date), h_l3_<stat> (last 3 games) and h_ps_<stat> (last season's per-game average, from prior_stats_df), with the game
    counts h_games and h_ps_games."""
    logs = _game_logs(stats_df, name_col) if stats_df is not None and len(stats_df) else _game_logs_empty()
    logs = logs[logs['week'] < int(week)].sort_values(['_key', 'pos', 'week'])
    g = logs.groupby(['_key', 'pos'])
    parts = [g[list(PROP_MARKETS)].mean().add_prefix('h_s2d_'),
             g[list(PROP_MARKETS)].apply(lambda x: x.tail(3).mean()).add_prefix('h_l3_'),
             g.size().rename('h_games')]
    if prior_stats_df is not None and len(prior_stats_df):
        prev = _game_logs(prior_stats_df, prior_name_col or name_col)
        gp = prev.groupby(['_key', 'pos'])
        parts += [gp[list(PROP_MARKETS)].mean().add_prefix('h_ps_'), gp.size().rename('h_ps_games')]
    return pd.concat(parts, axis=1)


def _game_logs_empty():
    return pd.DataFrame(columns=['_key', 'pos', 'week', *PROP_MARKETS]).astype({'week': int})


def recalibrated_mean(stat, pos, projection, s2d=None, l3=None, ps=None):
    """The recalibrated mean for each row (arrays). A position the engine has no fit for keeps its projection."""
    eng = load_engines().get(stat)
    m = np.asarray(projection, dtype=float)
    if not eng:
        return m.copy()
    pos = np.asarray(pos).astype(str)
    hist = [np.full(len(m), np.nan) if v is None else np.asarray(v, dtype=float) for v in (s2d, l3, ps)]
    Z = np.column_stack([m] + [np.where(np.isnan(h), m, h) for h in hist] + [np.isnan(h).astype(float) for h in hist])
    out = m.copy()
    for ps_, fit in eng['recal'].items():
        sel = pos == ps_
        if sel.any():
            out[sel] = np.maximum(Z[sel] @ np.asarray(fit['coef']) + fit['intercept'], fit['floor'])
    return out


def prob_over(stat, pos, mean, line):
    """P(result > line) for each row: the engine's empirical ratio (actual / mean) at the mean's level, read at the line. A
    count is read as spread over [k - 0.5, k + 0.5], so an x.5 line is exact and a whole-number line counts a push as not over.
    NaN where the engine has no table for the position."""
    eng = load_engines().get(stat)
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-6)
    line = np.asarray(line, dtype=float)
    pos = np.asarray(pos).astype(str)
    out = np.full(len(mean), np.nan)
    if not eng:
        return out
    probs = np.linspace(0, 1, 201)
    for ps_, tab in eng['M1'].items():
        sel = np.where(pos == ps_)[0]
        if not len(sel):
            continue
        edges = np.asarray(tab['edges'])
        b = np.searchsorted(edges, np.log(mean[sel]), side='right')
        for i, row in zip(sel, b):
            q = np.asarray(tab['ratio_quantiles'].get(str(int(row))) or next(iter(tab['ratio_quantiles'].values())))
            q = q + np.arange(len(q)) * 1e-12                      # ties (all the zero results) kept in order
            thr = (np.floor(line[i]) + 0.5) / mean[i]               # P(mean x ratio < floor(line) + 0.5) = P(result <= line)
            out[i] = 1.0 - float(np.interp(thr, q, probs))
    return out


def prop_probabilities(frame, stat, line_col='line', history=None):
    """`frame` (one row per player: Pos, _key, the board's `stat` projection and `line_col`) with 'prop_mean' (recalibrated)
    and 'p_over' at that line. `history` from player_history(); None leaves the history inputs missing.

    The engine was fit on the board's stat BEFORE the board flag 'v2_stat_recal_own_history' re-predicted it from the player's
    own history, so a board built with that flag (it keeps the earlier value in '_pre_recal_<stat>') is read from that column:
    the engine's own recalibration does the same job and must not run on an already recal'd stat."""
    out = frame.copy()
    proj = pd.to_numeric(out[stat], errors='coerce')
    pre = f'_pre_recal_{stat}'
    if pre in out.columns:
        proj = pd.to_numeric(out[pre], errors='coerce').fillna(proj)
    proj = proj.to_numpy(float)
    if history is not None and len(history):
        idx = pd.MultiIndex.from_arrays([out['_key'], out['Pos']])
        h = {k: history[f'h_{k}_{stat}'].reindex(idx).to_numpy(float) if f'h_{k}_{stat}' in history.columns else None
             for k in ('s2d', 'l3', 'ps')}
    else:
        h = {'s2d': None, 'l3': None, 'ps': None}
    out['prop_mean'] = recalibrated_mean(stat, out['Pos'], proj, h['s2d'], h['l3'], h['ps'])
    out['p_over'] = prob_over(stat, out['Pos'], out['prop_mean'], pd.to_numeric(out[line_col], errors='coerce'))
    return out
