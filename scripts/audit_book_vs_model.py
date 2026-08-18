"""
Audit: this app's own model vs the sportsbook/DFS consensus, at Half-PPR and
Full PPR, with a focus on receptions/receiving yards - the stat most
sensitive to a PPR setting and the one most likely to disagree with a market
that (per book) never even prices receptions as its own line (see
data.odds_projections's module docstring on Underdog's receiving-yards-only
board).

Run with the book payloads already saved to external_data/ (see
data.odds_sources.SAVED_PAYLOADS) - upload them once in Draft HQ, or drop
them in by hand. Reports, at both PPR settings:

  1. Rank correlation and median bias, our Proj Pts vs the market's Book
     Proj (book where it spoke, us where it didn't - the number actually
     comparable to a full projection).
  2. Same, restricted to the narrow like-for-like scope each market line
     actually covers (Ours (matched) vs Market Pts) - the honest read on
     whether the MODEL disagrees with the market, not just whether a
     receptions gap moves the total.
  3. Receptions specifically: our projected receptions vs the book
     consensus's own receptions line, for every player at least one book
     actually priced a receptions market for (DraftKings is the only book
     of the five that posts one - see BOOK_WEIGHTS).
  4. A sanity block (no negative projections, Ceiling >= Proj >= Floor,
     no NaN weighted lines) - the same pattern-worth-reusing HANDOFF.md
     section 6 describes for Draft HQ audits generally.

No network required beyond the initial FantasyPros ECR/ADP pull (which is
already cached for 6h by data.draft_sources). Book data is read from
external_data/*.json, never fetched live, so this is repeatable offline
once the payloads are saved once.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from data.draft_sources import load_ecr_raw, build_ecr_board, fetch_adp
from data.draft_projections import build_projected_board, PROJECTED_STATS
from data.draft_board import DEFAULT_SCORING, build_draft_board
import data.odds_sources as os_
from data.odds_projections import market_stat_lines, score_market_lines, compare_to_board

LATEST_SEASON = 2025


def _load_all_book_props():
    frames = []
    parsers = {
        'Underdog': lambda p: os_.parse_underdog_payload(p),
        'PrizePicks': lambda p: os_.parse_prizepicks_payload(p),
        'DraftKings': lambda p: os_.parse_draftkings_payloads(p),
        'FanDuel': lambda p: os_.parse_fanduel_payload(p),
        'Pinnacle': lambda p: os_.parse_pinnacle_payload(p),
    }
    status = {}
    for provider in os_.BOOKS:
        payload = os_.load_saved_book_payload(provider)
        if payload is None:
            status[provider] = 'no saved payload'
            continue
        props, err = parsers[provider](payload)
        status[provider] = f"{len(props)} rows" if not props.empty else (err or 'empty')
        if not props.empty:
            frames.append(props)
    return os_.combine_props(*frames), status


def _scoring_for(ppr):
    scoring = dict(DEFAULT_SCORING)
    scoring['rec'] = ppr
    return scoring


def _build_board(scoring):
    raw, err = load_ecr_raw()
    if raw is None or raw.empty:
        raise SystemExit(f"Couldn't load ECR: {err}")
    ecr_board = build_ecr_board(raw, 'Redraft 1QB')
    projected, _ = build_projected_board(ecr_board, scoring, latest_season=LATEST_SEASON)
    adp_df, _ = fetch_adp('Full PPR' if scoring['rec'] >= 0.75 else 'Half-PPR',
                          12, False, LATEST_SEASON)
    settings = {'scoring': scoring, 'roster': {}, 'num_teams': 12, 'baseline_season': LATEST_SEASON}
    board, _ = build_draft_board(projected, settings, adp_df=adp_df, latest_season=LATEST_SEASON)
    return board


def _spearman(a, b):
    a = pd.Series(a).rank()
    b = pd.Series(b).rank()
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return float('nan')
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _report_ppr_level(label, ppr, combined_props):
    print(f"\n{'=' * 70}\n{label} (rec={ppr})\n{'=' * 70}")
    scoring = _scoring_for(ppr)
    board = _build_board(scoring)
    print(f"Board: {len(board)} players, Proj Pts range "
          f"[{board['Proj Pts'].min():.1f}, {board['Proj Pts'].max():.1f}]")

    market_rows = market_stat_lines(combined_props, season_only=True, board=board)
    market_scored = score_market_lines(market_rows, scoring, positions=None)
    comparison, meta = compare_to_board(board, market_scored, scoring)
    print(f"Market join: {meta}")

    if not comparison.empty:
        corr_book = _spearman(comparison['Proj Pts'], comparison['Book Proj'])
        corr_matched = _spearman(comparison['Ours (matched)'], comparison['Market Pts'])
        bias_book = (comparison['Proj Pts'] - comparison['Book Proj']).median()
        print(f"Rank-corr (Proj Pts vs Book Proj, full-scope): {corr_book:.3f}")
        print(f"Rank-corr (matched-scope model vs market):     {corr_matched:.3f}")
        print(f"Median bias (Proj Pts - Book Proj):             {bias_book:+.1f}")
        for pos in ('QB', 'RB', 'WR', 'TE'):
            sub = comparison[comparison['Pos'] == pos]
            if len(sub) >= 3:
                b = (sub['Proj Pts'] - sub['Book Proj']).median()
                print(f"  {pos}: n={len(sub):3d}  median bias {b:+6.1f}  "
                     f"mean |Edge %| {sub['Edge %'].abs().mean():5.1f}%")

    # Receptions specifically - the ask.
    if 'receptions' in market_rows.columns:
        rec_book = market_rows[['player_key', 'player', 'position', 'receptions', 'Books']].dropna(
            subset=['receptions'])
        rec_book = rec_book[rec_book['position'].isin(['WR', 'TE', 'RB'])]
        board_rec = board[['Player', 'Pos', 'receptions', 'Proj Pts']].copy()
        board_rec['player_key'] = board_rec['Player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
        rec_book['pk'] = rec_book['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
        merged = rec_book.merge(board_rec, left_on='pk', right_on='player_key', how='inner',
                                suffixes=('_book', '_ours'))
        if not merged.empty:
            merged['delta'] = merged['receptions_ours'] if 'receptions_ours' in merged.columns \
                else merged['receptions']
            our_col = 'receptions' if 'receptions_ours' not in merged.columns else 'receptions_ours'
            book_col = 'receptions_book' if 'receptions_book' in merged.columns else 'receptions'
            merged['rec_delta'] = merged[our_col] - merged[book_col]
            print(f"\nReceptions: our projection vs book consensus line, n={len(merged)}")
            print(f"  Median our={merged[our_col].median():.1f}  book={merged[book_col].median():.1f}  "
                 f"delta(ours-book)={merged['rec_delta'].median():+.1f}")
            top = merged.reindex(merged['rec_delta'].abs().sort_values(ascending=False).index).head(8)
            cols = ['player', 'position', book_col, our_col, 'rec_delta']
            print(top[[c for c in cols if c in top.columns]].to_string(index=False))
    return board, comparison


def _sanity_checks(board):
    print("\n--- Sanity checks ---")
    checks = []
    checks.append(('No negative Proj Pts', (board['Proj Pts'] >= 0).all()))
    if 'Ceiling' in board.columns and 'Floor' in board.columns:
        checks.append(('Ceiling >= Proj Pts', (board['Ceiling'] >= board['Proj Pts'] - 0.05).all()))
        checks.append(('Floor <= Proj Pts', (board['Floor'] <= board['Proj Pts'] + 0.05).all()))
    if 'Big Play Pts' in board.columns:
        checks.append(('Big Play Pts finite, no NaN corrupting Proj Pts',
                       board['Proj Pts'].notna().all()))
    for stat in PROJECTED_STATS:
        if stat in board.columns:
            bad = board[stat] < -0.01
            if bad.any():
                checks.append((f'{stat} has {int(bad.sum())} negative rows', False))
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main():
    combined_props, status = _load_all_book_props()
    print("Book payloads loaded:")
    for provider, s in status.items():
        print(f"  {provider}: {s}")
    print(f"Combined scorable season props: "
         f"{int(combined_props['scorable'].sum()) if not combined_props.empty else 0}")
    print(f"Weights in use: {os_.BOOK_WEIGHTS}")

    board_half, _ = _report_ppr_level('HALF-PPR', 0.5, combined_props)
    board_full, _ = _report_ppr_level('FULL PPR', 1.0, combined_props)

    ok = _sanity_checks(board_full) and _sanity_checks(board_half)
    print(f"\n{'ALL SANITY CHECKS PASSED' if ok else 'SOME SANITY CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
