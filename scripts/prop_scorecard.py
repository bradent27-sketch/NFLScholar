"""
The weekly prop record (2026-10-28), plus the prop sheet for an upcoming week. data/prop_probability.py's P(over) on the live boards
is scored against the books' de-vigged price and the result.

Record: each player-stat on a pre-kickoff ledger board (data/ledger_scorecard.py) with a priced line, taken from the sharpest book
whose pulls carry a de-vigged p_over (Pinnacle, then DraftKings). Lines come from external_data/props_archive at two times: the
line posted when the board was built (what could have been bet then) and the last line before kickoff (the close). For each:
  - Brier of the model's P(over) and of the book's, the over rate, and calibration (the model's P(over) in bands against the hit
    rate);
  - the shrink toward 1/2 that minimises the model's Brier (p' = 1/2 + s (p - 1/2); 1 = trust as is, 0 = no information);
  - the record betting the model's side where its P(over) differs from the book's price by 3%, 5%, 8% or 12%: hit rate, 95% CI,
    and ROI at -110 and at the book's fair price.
  - (2026-11-11, audit B3) the same bets at the price posted for the side taken; three blind baselines per stat (always under,
    always over, the side the price favours) and the model's lift over the best; the model's side at every book's own line and
    price; overs and unders separately. The scoring lives in data/prop_record.py, shared with the Track Record page.
  - (2026-11-11, audit B6) the tiers' bets against the close: line value and price value (data/prop_tiers.clv_values).
Yardage lines under 5 are left out, as the app does. Pushes are out.

Sheet (--sheet WEEK): the latest ledger board for WEEK against the latest priced lines: the recalibrated mean, P(over), the book's
price, the gap, and the P(over) shrunk by the record's closing-line factor. A gap is a lean to track until the record says otherwise.

    python scripts/prop_scorecard.py [--year 2026] [--out .sweeps/prop_scorecard.txt]
    python scripts/prop_scorecard.py --sheet 9 [--year 2026]
    python scripts/prop_scorecard.py --install-engines [--engines-from .sweeps/deep_audit/prop_models_v2]
"""
import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ledger_scorecard as sc  # noqa: E402
from data import prop_probability as pp  # noqa: E402
from data import prop_record as pr  # noqa: E402
from data import prop_tiers as ptiers  # noqa: E402
from data.loaders import load_schedule  # noqa: E402
from data.prop_record import MIN_LEAN_LINE, best_shrink, load_archive, priced_lines, record_tables  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
# the closing-line shrink measured by the deep audit on 2026 weeks 3-7 (today's model rebuilt), until the record has its own
DEFAULT_SHRINK = 0.34


def install_engines(src):
    engines = {}
    for path in sorted(glob.glob(os.path.join(src, '*.json'))):
        with open(path, encoding='utf-8') as fh:
            eng = json.load(fh)
        engines[eng['stat']] = {k: eng[k] for k in ('spec', 'seasons', 'recal', 'M1', 'mean') if k in eng}
    with open(pp.ENGINE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(engines, fh, separators=(',', ':'))
    print(f'{len(engines)} engines -> {pp.ENGINE_PATH}: {", ".join(engines)}')


def load_inputs(year):
    stats_df, _team_col, name_col, _ = load_and_merge_data(year, 'Full PPR')
    try:
        prior, _t, prior_name_col, _ = load_and_merge_data(year - 1, 'Full PPR')
    except Exception:
        prior, prior_name_col = None, None
    return stats_df, name_col, prior, prior_name_col


def run_record(a):
    stats_df, name_col, prior, prior_name_col = load_inputs(a.year)
    rows, info = pr.build_record(a.year, stats_df, name_col, prior, prior_name_col, load_schedule(a.year))
    summary, bets, bands = record_tables(rows)
    sharp = rows[rows['sharpest']] if len(rows) else rows
    fmt = lambda v: f'{v:.3f}'  # noqa: E731
    skipped = info.get('skipped_files') or []
    out = [f"{a.year} prop record: data/prop_probability.py's P(over) on the pre-kickoff boards against the priced posted lines "
           f"(sharpest book: Pinnacle, then DraftKings, ...), weeks {info.get('weeks', [])}, {len(sharp):,} line-results "
           f"({len(rows):,} with every book's line)" + (f"; {len(skipped)} unreadable snapshot(s) skipped" if skipped else ''), '',
           '1. BRIER (lower is better; 0.25 = a coin flip) and the best shrink toward 1/2 per stat (1 = trust the model as is)',
           summary.to_string(index=False, float_format=fmt), '',
           "2. CALIBRATION, all stats: the model's P(over) in bands against the hit rate",
           bands.to_string(index=False, float_format=fmt), '',
           "3. BETTING THE MODEL'S SIDE where its P(over) differs from the book's de-vigged price by at least edge_min "
           "(a -110 bet needs 52.4%); roi_price = at the price posted for the side taken (priced = bets that had one)",
           bets.to_string(index=False, float_format=fmt), '',
           "4. BASELINES, sharpest book, every line: always under / always over / the side the book's price favours, against "
           "the model's side (all lines, and at 5%+); lift = model minus the best blind strategy. ROI at the posted price.",
           pr.baseline_table(rows).to_string(index=False, float_format=fmt), '',
           "5. BY BOOK: the model's side at each book's own line and price, 5%+ from that book's de-vigged price (every book "
           "that posted a priced line, not only the sharpest)",
           pr.book_table(rows).to_string(index=False, float_format=fmt), '',
           "6. BY SIDE, sharpest book: the model's overs and unders, every line and 5%+",
           pr.side_table(rows).to_string(index=False, float_format=fmt)]
    fwd = ptiers.forward_rows(rows)
    if len(fwd):
        out += ['', "7. BEATING THE CLOSE (B6), the tiers' bets at the board: line value = points the same book's last line "
                "before kickoff moved toward the side taken; price value = the sharpest book's closing fair chance of the side "
                "at the bet's line minus the chance the price paid implied (about -0.03 = the vig alone). 'beats close' needs "
                f"{ptiers.CLV_MIN_BETS}+ bets with the 90% low above 0.",
                ptiers.clv_table(fwd, ('tier',)).to_string(index=False, float_format=fmt), '',
                ptiers.clv_table(fwd, ('stat', 'side')).to_string(index=False, float_format=fmt)]
    out += ['', level_check(a.year, rows, stats_df, name_col)]
    text = '\n'.join(out)
    print(text)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    rows.to_csv(a.csv, index=False)


A6_FROM_WEEK = 9          # the first week on the recal and capacity-relief stat columns


def level_check(year, rows, stats_df, name_col, from_week=A6_FROM_WEEK):
    """A6 (2026-11-11): per stat, the model's mean P(over) at the sharpest book's closing line against the share that went over,
    and the share of played results above the model's median (ledger 'Median <stat>', every pre-kickoff row with a result),
    for weeks before `from_week` (early) and from it on (late). Bar to recentre a median (MEDIAN_CURVES_RECAL): weeks 9-12 off
    by 3+ points on 400+ lines, the same sign as weeks 3-8."""
    lines = ['8. LEVEL CHECK (A6): mean model P(over) against the over rate at the close, and the share of results above the '
             f'model median; weeks {from_week}+ (late) against the weeks before (early). No change before week 12 has results.']
    r = rows[(rows['timing'] == 'at close') & rows['p_over'].notna()]
    if 'sharpest' in r.columns:
        r = r[r['sharpest'].astype(bool)]
    frame, _info = sc.load_scored_ledger(year, stats_df, name_col, load_schedule(year))
    frame = frame[frame['pre_kickoff']] if 'pre_kickoff' in frame.columns else frame
    try:     # boards from before the ledger stored medians (D1, 2026-11-10): the same medians from the stored means
        from data.weekly_median import attach_medians
        filled = attach_medians(frame)
        for c in [c for c in filled.columns if str(c).startswith('Median ')]:
            frame[c] = frame[c].fillna(filled[c]) if c in frame.columns else filled[c]
    except Exception:
        pass
    out = []
    for stat in pp.PROP_MARKETS:
        rec = {'stat': stat}
        for label in ('early', 'late'):
            g = r[(r['stat'] == stat) & ((r['week'] < from_week) if label == 'early' else (r['week'] >= from_week))]
            rec[f'{label} lines'] = len(g)
            rec[f'{label} mean P'] = g['p_over'].mean() if len(g) else np.nan
            rec[f'{label} over rate'] = g['over'].mean() if len(g) else np.nan
        med, act = f'Median {stat}', f'actual_{stat}'
        if med in frame.columns and act in frame.columns:
            f = frame[pd.to_numeric(frame[med], errors='coerce').gt(0) & frame[act].notna()]
            for label in ('early', 'late'):
                g = f[(f['week'] < from_week) if label == 'early' else (f['week'] >= from_week)]
                rec[f'{label} above median'] = (g[act] > g[med]).mean() if len(g) else np.nan
        out.append(rec)
    t = pd.DataFrame(out)
    t['early gap'] = t['early over rate'] - t['early mean P']
    t['late gap'] = t['late over rate'] - t['late mean P']
    lines.append(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    return '\n'.join(lines)


def run_sheet(a):
    stats_df, name_col, prior, prior_name_col = load_inputs(a.year)
    paths = sorted(glob.glob(os.path.join(sc.LEDGER_DIR, f'{a.year}_wk{int(a.sheet):02d}_*.parquet')))
    if not paths:
        print(f'no ledger board for {a.year} week {a.sheet}')
        return
    board = pd.read_parquet(paths[-1])
    board['_key'] = clean_name_exact(board['Player'])
    board['week'] = int(a.sheet)
    arc = load_archive(a.year)
    arc = arc[arc['week'] == int(a.sheet)]
    now = pd.Timestamp.now(tz='UTC')
    lines = priced_lines(arc, {(int(a.sheet), k): now for k in board['_key']})
    hist = pp.player_history(stats_df, name_col, int(a.sheet), prior, prior_name_col)
    shrink = DEFAULT_SHRINK
    if os.path.exists(a.csv):
        rec = pd.read_csv(a.csv)
        rec = rec[(rec['timing'] == 'at close') & rec['p_over'].notna()]
        if 'sharpest' in rec.columns:
            rec = rec[rec['sharpest'].astype(bool)]
        if len(rec) >= 300:
            shrink = best_shrink(rec['p_over'].to_numpy(), rec['over'].to_numpy())
    rows = []
    for stat, market in pp.PROP_MARKETS.items():
        if stat not in board.columns:
            continue
        pre = [f'_pre_recal_{stat}'] if f'_pre_recal_{stat}' in board.columns else []
        f = board[['Player', 'Pos', 'Team', '_key', stat] + pre].copy()
        got = [lines.get((int(a.sheet), k, market)) for k in f['_key']]
        f['line'] = [g[0] if g else np.nan for g in got]
        f['p_book'] = [g[1] if g else np.nan for g in got]
        f['book'] = [g[2] if g else None for g in got]
        f = f[f['line'].notna() & pd.to_numeric(f[stat], errors='coerce').gt(0)]
        f = f[f['line'] >= MIN_LEAN_LINE.get(stat, 0.0)]
        if f.empty:
            continue
        f = pp.prop_probabilities(f, stat, history=hist)
        f['stat'] = stat
        rows.append(f.rename(columns={stat: 'projection'}))
    if not rows:
        print('no priced lines for that week')
        return
    sheet = pd.concat(rows, ignore_index=True)
    sheet['p_over_shrunk'] = 0.5 + shrink * (sheet['p_over'] - 0.5)
    sheet['gap'] = sheet['p_over_shrunk'] - sheet['p_book']
    sheet = sheet.sort_values('gap', key=np.abs, ascending=False)
    path = os.path.join('.sweeps', f'prop_sheet_{a.year}_wk{int(a.sheet):02d}.csv')
    sheet.to_csv(path, index=False)
    print(f'board {os.path.basename(paths[-1])}; {len(sheet)} priced lines; shrink {shrink:.2f}; -> {path}')
    print(sheet.head(25)[['Player', 'Pos', 'stat', 'line', 'book', 'projection', 'prop_mean', 'p_over', 'p_over_shrunk', 'p_book',
                          'gap']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--out', default='.sweeps/prop_scorecard.txt')
    ap.add_argument('--csv', default='.sweeps/prop_scorecard_rows.csv')
    ap.add_argument('--sheet', type=int, default=None, help='write the prop sheet for this week instead of the record')
    ap.add_argument('--engines', default=None, help='score with this engine file instead of data/prop_engines.json')
    ap.add_argument('--install-engines', action='store_true')
    ap.add_argument('--engines-from', default=os.path.join('.sweeps', 'deep_audit', 'prop_models_v2'))
    a = ap.parse_args()
    if a.engines:
        pp.ENGINE_PATH = a.engines
    if a.install_engines:
        install_engines(a.engines_from)
    elif a.sheet is not None:
        run_sheet(a)
    else:
        run_record(a)


if __name__ == '__main__':
    main()
