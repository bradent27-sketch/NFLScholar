"""
Prop engine refit on the shipped stat columns (2026-11-01, bar in .sweeps/prop_engine_refit_bar.md, set before the run).

data/prop_engines.json was fit on the board's stat columns before v2_stat_recal_own_history / _game_script re-predicted them,
and data/prop_probability.py reads '_pre_recal_<stat>' so the engine's own recalibration does not run twice. Here each stat is
scored three ways on the deep-audit panel, with the shipped columns rebuilt on it (v2_rb_game_script_rush, then the shipped
recal table):
  E0  the pre-recal column + the engine's linear recalibration (today's engine)
  E1  the shipped column + the engine's linear recalibration
  E2  the shipped column as the mean
each with the empirical spread (M1), forward-chained 2023-2026 (CRPS), and on 2026's posted lines (Brier, side record).

    python scripts/eval_prop_engine_refit.py
    python scripts/eval_prop_engine_refit.py --install passing_yards,receiving_yards   # E2 engines into data/prop_engines.json

--install fits the named stats' E2 engines on every panel season (no recalibration, the empirical tables on the shipped
column) and marks them "mean": "board", so data/prop_probability.py reads the board's stat instead of '_pre_recal_<stat>'.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data.weekly_projections as wp  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from scripts.deep_audit import (ENGINE_RAW_MEAN, PROP_SPECS, SEED, dist_cdfs, lin_means, load_panel, p_over_at,  # noqa: E402
                                posted_join, posted_lines_2026, prop_rows, score_cdf)

OUT = '.sweeps/prop_refit'
UNMOVED = ('passing_attempts', 'passing_completions')     # the recal leaves these alone: E1 = E0
FEATS = {'v2_stat_recal_own_history', 'v2_stat_recal_own_history_edges', 'v2_stat_recal_game_script'}
GAP = 0.05


def week_scripts(p):
    """{(year, week): {team: (m, t)}} from the panel's implied points (stat_recal_script_inputs' definition)."""
    teams = p[['year', 'week', 'team', 'imp_team', 'imp_opp']].dropna().drop_duplicates(['year', 'week', 'team'])
    teams = teams.join(teams.groupby(['year', 'week'])['imp_team'].mean().rename('league'), on=['year', 'week'])
    return {k: {str(r.team): (float(r.imp_team - r.imp_opp), float(r.imp_team - r.league)) for r in g.itertuples()}
            for k, g in teams.groupby(['year', 'week'])}


def shipped_columns(p):
    """p with s_<stat>: the board's stat as today's DEFAULT_FEATURES leave it - v2_rb_game_script_rush's multiplier on RB
    rushing yards / TDs (weeks 3+), then the shipped recal (own history weeks 1-17, game script for QBs weeks 3-17)."""
    stats = sorted({s for v in wp.STAT_RECAL_OWN_HISTORY.values() for s in v} | set(wp.STAT_RECAL_GAME_SCRIPT['QB'])
                   | set(wp.RB_GAME_SCRIPT_RUSH))
    p = p.copy()
    for s in stats:
        p[f's_{s}'] = pd.to_numeric(p[f'p_{s}'], errors='coerce')
    scripts = week_scripts(p)
    rb = (p['pos'] == 'RB') & (p['week'] >= 3)
    for (yr, wk), idx in p[rb].groupby(['year', 'week']).groups.items():
        mult = wp.rb_game_script_multipliers(p.loc[idx, 'team'].astype(str).to_numpy(), scripts.get((yr, wk), {}),
                                             wp.RB_GAME_SCRIPT_RUSH)
        for s, f in mult.items():
            p.loc[idx, f's_{s}'] = p.loc[idx, f's_{s}'].to_numpy(float) * f
    hcols = ['h_games', 'h_ps_games'] + [f'h_{k}_{s}' for s in stats for k in ('s2d', 'l3', 'ps')]
    for (yr, wk), g in p.groupby(['year', 'week']):
        form, table = wp.stat_recal_table(FEATS, wk)
        if not table:
            continue
        res = pd.DataFrame({'Player': g['player'].values, 'Pos': g['pos'].values, 'Team': g['team'].values,
                            'Model Proj Pts': g['p_points'].values, 'Availability': g['Availability'].values,
                            'Workload If Active': g['Workload If Active'].values,
                            'QB Projected Starter': g['QB Projected Starter'].values})
        for s in stats:
            res[s] = g[f's_{s}'].values
        hist = g[hcols].copy()
        hist.index = pd.MultiIndex.from_arrays([clean_name_exact(g['player'].astype(str)).values, g['pos'].values])
        hist = hist[~hist.index.duplicated(keep=False)]
        new, _ = wp.apply_stat_recal_own_history(res, hist, form, table, script=scripts.get((yr, wk), {}))
        for s in stats:
            if f'_pre_recal_{s}' in new.columns:
                p.loc[g.index, f's_{s}'] = pd.to_numeric(new[s], errors='coerce').to_numpy(float)
    return p


def install(stats):
    """E2 engines for `stats`, fit on every panel season, merged into data/prop_engines.json."""
    import json
    import tempfile
    from data import prop_probability as pp
    from scripts.deep_audit import save_prop_engine
    p = shipped_columns(load_panel())
    with open(pp.ENGINE_PATH, encoding='utf-8') as fh:
        engines = json.load(fh)
    with tempfile.TemporaryDirectory() as tmp:
        for stat in stats:
            spec = PROP_SPECS[stat]
            d = prop_rows(p, stat)
            floor = 0.05 if spec['kind'] == 'count' else 1.0
            save_prop_engine(d, stat, spec, d[f's_{stat}'].to_numpy(float), d[f'y_{stat}'].to_numpy(float),
                             d['pos'].to_numpy(), floor, np.ones(len(d), dtype=bool), tmp, recal=False)
            with open(os.path.join(tmp, f'{stat}.json'), encoding='utf-8') as fh:
                eng = json.load(fh)
            engines[stat] = {**{k: eng[k] for k in ('spec', 'seasons', 'recal', 'M1')}, 'mean': 'board'}
            print(f'{stat}: E2 engine on {len(d):,} rows, seasons {eng["seasons"][0]}-{eng["seasons"][-1]}')
    with open(pp.ENGINE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(engines, fh, separators=(',', ':'))
    print(f'-> {pp.ENGINE_PATH}')


def main():
    if '--install' in sys.argv:
        return install(sys.argv[sys.argv.index('--install') + 1].split(','))
    rng = np.random.default_rng(SEED + 7)
    p = shipped_columns(load_panel())
    arc, kick = posted_lines_2026()
    crps_rows, posted = [], []
    for stat, spec in PROP_SPECS.items():
        if stat in UNMOVED:
            continue
        d = prop_rows(p, stat)
        y, m0, ms = (d[f'y_{stat}'].to_numpy(float), d[f'p_{stat}'].to_numpy(float), d[f's_{stat}'].to_numpy(float))
        pos_arr, yr = d['pos'].to_numpy(), d['year'].to_numpy()
        floor = 0.05 if spec['kind'] == 'count' else 1.0
        print(f'{stat}: {len(d):,} rows, shipped column moved on {np.mean(np.abs(ms - m0) > 1e-9):.0%}', flush=True)
        for t in (2023, 2024, 2025, 2026):
            tr, te = yr < t, yr == t
            means = {'E0': m0 if stat in ENGINE_RAW_MEAN else lin_means(d, stat, m0, y, tr, pos_arr, floor)[0],
                     'E1': lin_means(d, stat, ms, y, tr, pos_arr, floor)[0],
                     'E2': ms}
            F = {k: dist_cdfs(v, y, tr, te, pos_arr, spec, spec['grid'])[0] for k, v in means.items()}
            for k, f in F.items():
                crps, _, _ = score_cdf(f, y[te], rng)
                crps_rows.append({'stat': stat, 'season': t, 'variant': k, 'n': int(te.sum()), 'crps': float(crps.mean()),
                                  'rmse': float(np.sqrt(np.mean((means[k][te] - y[te]) ** 2)))})
            if t == 2026:
                dt = d[te].reset_index(drop=True)
                for _, L in posted_join(dt, spec, arc, kick).iterrows():
                    i, line = int(L['i']), float(L['line'])
                    rec = {'stat': stat, 'week': int(L['week']), 'player': dt.at[i, 'player'], 'pos': dt.at[i, 'pos'],
                           'line': line, 'p_book': float(L['p_book']), 'actual': float(y[te][i])}
                    for k, f in F.items():
                        rec[f'p_{k}'] = float(p_over_at(f[i:i + 1], np.array([line]))[0])
                        rec[f'mean_{k}'] = float(means[k][te][i])
                    posted.append(rec)
    os.makedirs(OUT, exist_ok=True)
    c = pd.DataFrame(crps_rows)
    po = pd.DataFrame(posted)
    c.to_csv(os.path.join(OUT, 'crps.csv'), index=False)
    po.to_csv(os.path.join(OUT, 'posted_2026.csv'), index=False)

    print('\nforward-chained CRPS (2023-2026), change vs E0 in %, seasons better of 4; mean RMSE')
    verdicts = {}
    for stat, g in c.groupby('stat', sort=False):
        w = g.pivot_table(index='season', columns='variant', values='crps')
        n = g[g['variant'] == 'E0'].set_index('season')['n']
        pooled = {k: float((w[k] * n).sum() / n.sum()) for k in w.columns}
        rm = g.groupby('variant')['rmse'].mean()
        line = f'  {stat:22s} E0 {pooled["E0"]:.4f}'
        q = po[po['stat'] == stat]
        ok = {}
        for k in ('E1', 'E2'):
            ch = 100 * (pooled[k] / pooled['E0'] - 1)
            better = int((w[k] < w['E0']).sum())
            brier = {v: float(np.mean((q[f'p_{v}'] - (q['actual'] > q['line'])) ** 2)) if len(q) else np.nan
                     for v in ('E0', k)}
            ok[k] = ch < 0 and better >= 3 and (not len(q) or brier[k] - brier['E0'] <= 0.002)
            line += f' | {k} {ch:+.2f}% ({better}/4), Brier {brier[k] - brier["E0"]:+.4f}'
        line += f' | RMSE E0 {rm["E0"]:.3f} E1 {rm["E1"]:.3f} E2 {rm["E2"]:.3f}'
        ch1, ch2 = (100 * (pooled[k] / pooled['E0'] - 1) for k in ('E1', 'E2'))
        pick = ('E2' if ok['E2'] and (not ok['E1'] or ch2 <= ch1 + 0.1) else 'E1' if ok['E1'] else 'E0')
        verdicts[stat] = pick
        print(line + f' -> {pick}')

    print('\n2026 posted lines: Brier (book de-vigged / E0 / E1 / E2) and the side record at a 5%+ gap to the book')
    for stat, q in po.groupby('stat', sort=False):
        o = (q['actual'] > q['line']).astype(float)
        push = q['actual'] == q['line']
        parts = [f'n {len(q)}', f"book {np.mean((q['p_book'] - o) ** 2):.4f}"]
        for k in ('E0', 'E1', 'E2'):
            br = np.mean((q[f'p_{k}'] - o) ** 2)
            gap = (q[f'p_{k}'] - q['p_book']).abs() >= GAP
            side = np.where(q[f'p_{k}'] > q['p_book'], 1.0, 0.0)
            sel = gap & ~push
            win = float(np.mean(side[sel] == o[sel])) if sel.any() else np.nan
            parts.append(f'{k} {br:.4f} side {win:.3f} ({int(sel.sum())})')
        print(f'  {stat:22s} ' + ' | '.join(parts))
    print('\nverdicts:', verdicts)


if __name__ == '__main__':
    main()
