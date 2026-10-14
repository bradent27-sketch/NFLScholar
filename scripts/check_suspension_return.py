"""
End-to-end check for v2_suspension_return_full_role (2026-10-14): build the harness base-arm board for every 2021-2025
week where a zero-game RB/WR/TE had been suspended through the previous week, flag on and flag off, diff the whole board
and score every row the flag changes against the actual points.

    python scripts/check_suspension_return.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.historical_availability import load_weekly_rosters, suspended_before  # noqa: E402
from data.transforms import load_and_merge_data  # noqa: E402
from data.utils import clean_name_exact  # noqa: E402
from data.weekly_projections import DEFAULT_FEATURES, build_weekly_projections  # noqa: E402
from scripts.eval_weekly_model import _actual_points  # noqa: E402

FLAG = 'v2_suspension_return_full_role'
BASE = frozenset(DEFAULT_FEATURES | {'v2_historical_injury_replay', 'v2_historical_reserve_replay'}) - {FLAG}
WEEKS = {2022: (6, 7), 2023: (4, 5), 2025: (4,)}


def main():
    pd.set_option('display.width', 230)
    changed_all, other_changed = [], 0
    for year, weeks in WEEKS.items():
        stats, _t, name_col, _ = load_and_merge_data(year, 'Full PPR')
        for week in weeks:
            actual = _actual_points(stats, name_col, week, 'fantasy_points_ppr')
            boards = {}
            for arm, feats in (('off', BASE), ('on', BASE | {FLAG})):
                build_weekly_projections.clear()
                boards[arm], _ = build_weekly_projections(year, week, 'Full PPR', as_of_week=week, apply_injury=False,
                                                          features=feats)
            build_weekly_projections.clear()
            k = ['Player', 'Pos', 'Team']
            off, on = (boards[a].set_index(k) for a in ('off', 'on'))
            j = off[['Model Proj Pts', 'Expected Snap Share', 'targets', 'rushing_attempts']].join(
                on[['Model Proj Pts', 'Expected Snap Share', 'targets', 'rushing_attempts']], lsuffix='_off', rsuffix='_on', how='outer')
            ch = j[(j['Model Proj Pts_off'] - j['Model Proj Pts_on']).abs() > 0.005].reset_index()
            ch['actual'] = ch['Player'].map(actual)
            ch['year'], ch['week'] = year, week
            susp = suspended_before(year, week)
            ch['suspended_return'] = clean_name_exact(ch['Player'].astype(str)).isin(susp).to_numpy()
            print(f'\n{year} wk{week}: board rows {len(off)}, rows changed {len(ch)}, suspended-return candidates on the board '
                  f'{int(clean_name_exact(pd.Series(off.reset_index()["Player"].astype(str))).isin(susp).sum())}')
            print(ch[['Player', 'Pos', 'Team', 'suspended_return', 'Expected Snap Share_off', 'Expected Snap Share_on',
                      'Model Proj Pts_off', 'Model Proj Pts_on', 'actual']].round(2).to_string(index=False))
            changed_all.append(ch)
    c = pd.concat(changed_all, ignore_index=True)
    c = c[c['actual'].notna()]
    for label, sub in (('flagged players', c[c.suspended_return]), ('teammates (room conservation)', c[~c.suspended_return])):
        if sub.empty:
            continue
        e_off = (sub['Model Proj Pts_off'] - sub['actual']) ** 2
        e_on = (sub['Model Proj Pts_on'] - sub['actual']) ** 2
        print(f'\n{label}: n={len(sub)} played-rows changed | sum sq err off {e_off.sum():.1f} -> on {e_on.sum():.1f} | '
              f'closer on {(e_on < e_off).sum()} of {len(sub)}')
    e_off = (c['Model Proj Pts_off'] - c['actual']) ** 2
    e_on = (c['Model Proj Pts_on'] - c['actual']) ** 2
    print(f'ALL changed rows with a box score: n={len(c)} sum sq err {e_off.sum():.1f} -> {e_on.sum():.1f}')


if __name__ == '__main__':
    main()
