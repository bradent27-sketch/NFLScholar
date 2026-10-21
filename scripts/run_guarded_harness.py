"""
Run scripts/backtest_component.py with the snap-count guard of scripts/capture_stat_blend_traces.py installed first (disk cache
in .sweeps/snap_cache, retries, stop instead of running on degraded data - see that script and backlog D9). Same arguments:

    python scripts/run_guarded_harness.py --add <flag> --add-features ... --years 2022,2023,2024,2025 --weeks 3-17
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.capture_stat_blend_traces import install_snap_guard  # noqa: E402

if __name__ == '__main__':
    install_snap_guard()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_component.py')
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name='__main__')
