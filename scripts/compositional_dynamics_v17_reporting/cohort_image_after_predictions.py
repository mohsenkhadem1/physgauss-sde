"""Wait for sealed physical forecasts, then run reviewed image prediction only.

No scoring or target/reference readers are invoked. Existing outputs are never
resumed or overwritten by this one-shot operational launcher.
"""
import json
import subprocess
import time
from pathlib import Path

from scripts.compositional_dynamics_v17_reporting import cohort_appearance_inference as image
from scripts.compositional_dynamics_v17_reporting.cohort_recovery_evaluation import committed

ROOT = image.ROOT
COHORT = ROOT / 'runs/compositional-dynamics-v17/fresh-cohort'
REPORT = ROOT / 'reports/compositional-dynamics-v17/cohort-image-launch-status.json'


def main():
    bindings = json.loads((ROOT / 'reports/compositional-dynamics-v17/fresh-cohort-launch-bindings.json').read_text())
    runtime = json.loads((COHORT / 'pipeline/runtime.json').read_text())
    if runtime['bindings'] != bindings:
        raise ValueError('Pipeline bindings changed')
    if REPORT.exists() or image.BASE.exists() or image.original.PLAN.exists():
        raise ValueError('Preserve existing image launch/output')
    image.bound()  # Validate the reviewed inference wrapper and frozen sources.
    expected = [COHORT / lane / f'world-{w}' / suffix / 'prediction-commit.json'
                for w in range(4)
                for role in ('forecast-40mm', 'forecast-80mm')
                for lane, suffix in [('shared-forecast', role), ('native', 'forecasts/' + role)]]
    while not all(p.exists() for p in expected):
        if time.time() >= runtime['deadline_epoch'] + 30:
            raise TimeoutError('Original physical pipeline deadline exhausted')
        state = subprocess.check_output(['systemctl', '--user', 'show',
                    'compositional-v17-fresh-cohort-pipeline', '--property=ActiveState', '--value'], text=True).strip()
        if state not in ('active', 'activating', 'deactivating'):
            raise RuntimeError('Physical pipeline stopped before all forecast seals')
        time.sleep(20)
    committed(bindings)  # Full identity and original16/recovered8 seal checks.
    image.prepare(bindings)
    image.predict()
    module = image.bound()
    _, seal = module.prediction_gate()
    image.original.write(REPORT, dict(
        completed=True, launcher_sha256=image.original.sha(__file__),
        prediction_commit_sha256=image.original.sha(image.BASE / 'prediction-commit.json'),
        expected=seal['expected'], completed_images=sum(r['completed'] for r in seal['rows']),
        failed_images=sum(not r['completed'] for r in seal['rows']),
        reference_images_read=False, scoring_run=False,
        scope='One-shot launch of reviewed prepare/predict after all physical forecast seals.'))


if __name__ == '__main__':
    main()
