"""Materialize paper artifacts after complete frozen predictions, without tuning.

Runs independently of the interactive session. Manuscript prose and final PDF
review remain explicit authoring steps; this service never commits or pushes.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE, ROOT, OUT, within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.campaign_jobs import jobs
from scripts.phystwin.sequential_reference import write


def run(script, *args):
    within_deadline()
    command = [str(ROOT / '.venv/bin/python'),
               str(ROOT / 'scripts/position_ood' / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    seal = verify()
    declared = jobs(seal['settings'])
    state = BASE / 'publication-ready.json'
    completed = set()
    if state.exists():
        completed.update(json.loads(state.read_text()).get('completed_stages', []))
    while len(completed) < 2:
        within_deadline()
        if 'prediction' not in completed and all(Path(j['result']).exists() for j in declared):
            verify()
            run('publication.py')
            run('compute_report.py')
            run('qualitative.py', '--video')
            completed.add('prediction')
        active_path = OUT / 'transfer_test-active-results.json'
        if 'acquisition' not in completed and active_path.exists():
            active = json.loads(active_path.read_text())
            coverage = [OUT / f'{split}-extrapolated-coverage.json'
                        for split in ('test_id', 'test_material', 'test_environment')]
            if (active.get('complete') and active.get('matched_frame_ablation', {}).get('complete')
                    and all(p.exists() for p in coverage)):
                verify()
                run('publication_uncertainty.py')
                run('figure_active.py')
                run('compute_report.py')
                completed.add('acquisition')
        write(state, dict(completed_stages=sorted(completed),
                          expected_stages=['acquisition', 'prediction'],
                          updated_unix=time.time(),
                          note='Generated artifacts require manuscript integration and visual review.'))
        if len(completed) < 2:
            time.sleep(60)


if __name__ == '__main__':
    main()
