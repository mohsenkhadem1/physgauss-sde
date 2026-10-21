"""Single-GPU fixed TRAIN queue for the explicit zero-force diagnostic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from isaac_probe_pool_coordinator import atomic_json, utc
from isaac_calibrated_probes_v03 import validate_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('runs/gaussian-boundary-neural-v01'))
    p.add_argument('--output', type=Path, default=Path('runs/gaussian-campaign-v01/linearity-diagnostic'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--isaac-python', default='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'.driver.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    path = a.output/'execution.json'
    if path.exists():
        raise FileExistsError('Inspect completed/partial diagnostic before any resume')
    worker = Path(__file__).with_name('isaac_probe_linearity_diagnostic.py')
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    ids = ['t0001', 't0000', 't0002', 't0003']
    ledger = {'driver_pid': os.getpid(), 'started_utc': utc(), 'episode_ids': ids,
        'gpu': a.gpu, 'worker_sha256': digest, 'cases': [], 'completed': False}
    atomic_json(path, ledger)
    (a.output/'logs').mkdir(exist_ok=True)
    for eid in ids:
        if hashlib.sha256(worker.read_bytes()).hexdigest() != digest:
            raise ValueError('Diagnostic source changed')
        command = [a.isaac_python, '-u', str(worker), '--dataset', str(a.dataset),
                   '--output', str(a.output), '--episode', eid, '--gpu', str(a.gpu)]
        row = {'episode_id': eid, 'started_utc': utc(), 'command': command, 'state': 'running'}
        ledger['cases'].append(row)
        started = time.monotonic()
        with (a.output/'logs'/(eid+'.log')).open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'})
            row['pid'] = process.pid
            atomic_json(path, ledger)
            code = process.wait()
        row['exit_code'] = code
        if not code:
            try:
                audit = validate_case(a.output, eid)
                atomic_json(a.output/'private'/eid/'validation.json', audit)
                row['state'] = 'completed_verified'
            except Exception as error:
                row.update(state='failed_validation', error=repr(error))
        else:
            row['state'] = 'failed'
        row.update(elapsed_s=time.monotonic()-started, finished_utc=utc())
        atomic_json(path, ledger)
        print(json.dumps(row), flush=True)
    ledger.update(completed=all(r['state'] == 'completed_verified' for r in ledger['cases']), finished_utc=utc())
    atomic_json(path, ledger)
    return not ledger['completed']


if __name__ == '__main__':
    raise SystemExit(main())
