"""Re-derive a calibration_lab dump's season-anchor inputs (anchor_mean / anchor_games) with the shared-name defenders dropped,
as the build has done since 2026-10-26 (drop_shared_name_defenders). Dumps built before scripts/calibration_lab.py did the same
(2026-10-28) carry a namesake defender's games in those columns (A.J. Green 2021-22, Spencer Brown 2022, 2026 rookies); the
board's own numbers were right.

    python scripts/patch_dump_anchor_inputs.py .sweeps/calibration_lab_v8.csv [more dumps]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from data.silent_games import with_silent_games  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.weekly_projections import DEFAULT_FEATURES, _played_weeks_before, drop_shared_name_defenders  # noqa: E402


def patch(path):
    d = pd.read_csv(path, low_memory=False)
    changed = 0
    for year in sorted(d['year'].unique()):
        stats, team_col, name_col, _ = load_and_merge_data(int(year), 'Full PPR')
        src = with_silent_games(stats, int(year), name_col, team_col) if 'v2_silent_games' in DEFAULT_FEATURES else stats
        src = drop_shared_name_defenders(src, name_col)[0]
        for week in sorted(d.loc[d['year'] == year, 'week'].unique()):
            rows = (d['year'] == year) & (d['week'] == week)
            hist = _played_weeks_before(src, int(week))
            if hist.empty or 'fantasy_points' not in hist.columns:
                continue
            g = pd.to_numeric(hist['fantasy_points'], errors='coerce').groupby(hist[name_col]).agg(['mean', 'count'])
            mean = d.loc[rows, 'player'].map(g['mean'])
            games = d.loc[rows, 'player'].map(g['count']).fillna(0).astype(int)
            old_m = pd.to_numeric(d.loc[rows, 'anchor_mean'], errors='coerce')
            moved = ~((mean - old_m).abs().fillna(0) < 1e-9) | (mean.isna() != old_m.isna())
            changed += int(moved.sum())
            d.loc[rows, 'anchor_mean'] = mean
            d.loc[rows, 'anchor_games'] = games
    d.to_csv(path, index=False)
    print(f'{path}: {changed} rows re-derived')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        patch(p)
