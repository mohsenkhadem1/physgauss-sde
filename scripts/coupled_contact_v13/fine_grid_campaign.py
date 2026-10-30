"""Durable six-worker secondary study, within the original campaign deadline."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from scripts.coupled_contact_v13.campaign import remaining, write
from scripts.coupled_contact_v13.fine_grid_pipeline import BASE, PARENT_BASE, check_freeze


def stage(name, script, arguments=()):
    check_freeze()
    marker = BASE/'stages'/(name+'.json')
    if marker.exists() and json.loads(marker.read_text()).get('success'):
        return
    with (BASE/(name+'.log')).open('a') as log:
        result = subprocess.run(
            [str(ROOT/'runs/phystwin-env/bin/python'),
             str(ROOT/'scripts/coupled_contact_v13'/script), *arguments],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
            timeout=max(1, remaining()), check=False)
    write(marker, {'success': result.returncode == 0,
                   'returncode': result.returncode,
                   'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if result.returncode:
        raise RuntimeError('Secondary stage failed: '+name)


def wait_parent(split):
    marker = PARENT_BASE/'stages'/(split+'-evaluate.json')
    while True:
        check_freeze()
        if marker.exists():
            if not json.loads(marker.read_text()).get('success'):
                raise RuntimeError('Parent evaluation failed: '+split)
            return
        time.sleep(min(20, max(0, remaining())))


def main():
    check_freeze()
    BASE.mkdir(parents=True, exist_ok=True)
    stage('library', 'fine_grid_pipeline.py', ['library', '--workers', '6'])
    for split in ('development', 'reserved'):
        wait_parent(split)
        stage(split+'-evaluate', 'fine_grid_pipeline.py',
              ['evaluate', '--split', split, '--workers', '6'])
    stage('report', 'fine_grid_report.py')
    stage('verify', 'fine_grid_verify.py')
    stage('report-audited', 'fine_grid_report.py')
    stage('render', 'fine_grid_render.py')
    write(BASE/'complete.json', {
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'Secondary grid study and numerical audit complete; visual review, paper and publication remain'})


if __name__ == '__main__':
    main()
