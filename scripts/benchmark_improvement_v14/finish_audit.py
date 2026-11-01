"""User-authorized post-campaign verification; never launches fitting or forecasts.

The original campaign deadline and frozen sources remain unchanged. This entry
point grants only the selected read-only audit a one-hour execution window.
"""
import argparse
import importlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark_improvement_v14 import common


def main():
    parser = argparse.ArgumentParser()
    choices = {
        'softvt': 'independent_audit_b_v14',
        'phystwin': 'phystwin_independent_audit',
        'gaussian': 'gaussian_independent_audit',
    }
    parser.add_argument('audit', choices=choices)
    args = parser.parse_args()
    deadline = time.monotonic() + 3600

    def audit_remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise RuntimeError('Post-campaign audit time limit reached')
        return seconds

    common.remaining = audit_remaining
    module = importlib.import_module('scripts.benchmark_improvement_v14.' + choices[args.audit])
    module.main()


if __name__ == '__main__':
    main()
