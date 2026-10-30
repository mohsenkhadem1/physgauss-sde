"""Bounded descriptive analyses after completed production verification/videos.

No PDF is authored here: final visual review and PDF rendering remain explicit.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from scripts.coupled_contact_v13.campaign import remaining, write
from scripts.coupled_contact_v13.surface_pipeline import BASE, REPORT, implementation

PLAN = REPORT/'postanalysis-plan.json'


def guard(expected_plan_sha256=None):
    implementation().check_freeze()
    content = PLAN.read_bytes()
    if expected_plan_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_plan_sha256:
        raise RuntimeError('Postanalysis plan changed during execution')
    plan = json.loads(content)
    if hashlib.sha256((REPORT/'method-freeze.json').read_bytes()).hexdigest() != plan['frozen_method_sha256']:
        raise RuntimeError('Original method freeze changed before postanalysis')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Postanalysis source changed: '+name)
    return plan


def main():
    plan_sha256 = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    plan = guard(plan_sha256)
    while not (BASE/'postprocessing-complete.json').exists():
        if remaining() <= 0:
            raise RuntimeError('Original deadline reached before postanalysis')
        time.sleep(min(20, remaining()))
    python = ROOT/'runs/phystwin-env/bin/python'
    for stage in plan['stages']:
        guard(plan_sha256)
        if remaining() <= 0:
            raise RuntimeError('Original deadline reached during postanalysis')
        with (BASE/('analysis-'+stage['name']+'.log')).open('w') as log:
            result = subprocess.run([str(python), str(ROOT/stage['script']), *stage['args']],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, remaining()))
        if result.returncode:
            write(BASE/'analysis-failed.json', dict(stage=stage['name'], returncode=result.returncode))
            raise RuntimeError('Postanalysis failed: '+stage['name'])
    guard(plan_sha256)
    write(BASE/'analysis-complete.json', dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=plan_sha256,
        scope='Descriptive analyses complete; independent result/visual review, PDF creation and publication remain required'))


if __name__ == '__main__':
    main()
