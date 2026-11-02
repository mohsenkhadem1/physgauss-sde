"""Bounded reporting watcher; never restarts a fit or modifies primary results."""
import json
import subprocess
import sys
import time
from scripts.compositional_neural_mechanics_reporting.discrepancy_band_score import B, R, AUDIT, read, passed


def main():
    deadline = read(B/'stage-clock.json')['deadline_unix']+180
    while time.time() < deadline:
        if AUDIT.exists():
            audit = read(AUDIT)
            if not passed(audit):
                raise RuntimeError('Primary independent audit failed; supplement not scored')
            subprocess.run([sys.executable, '-m',
                            'scripts.compositional_neural_mechanics_reporting.discrepancy_band_score',
                            '--score'], check=True)
            return
        (R/'discrepancy-band-watch-status.json').write_text(json.dumps(
            dict(status='waiting_for_primary_independent_audit', checked_unix=time.time(),
                 existing_deadline_unix=deadline), indent=2)+'\n')
        time.sleep(min(60, max(0, deadline-time.time())))
    raise TimeoutError('Existing study deadline elapsed; no supplement scored')


if __name__ == '__main__':
    main()
