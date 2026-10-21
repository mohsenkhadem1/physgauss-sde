"""Fixed TRAIN queue for the frozen force-angle curriculum; retain failures."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from isaac_probe_pool_coordinator import atomic_json, run_fixed_queue, utc
from isaac_calibrated_probes_v03 import validate_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/gaussian_rich_controls_v01.json'))
    p.add_argument('--isaac-python', default='/home/bayespc2/vscode/deformable worldmodel/.venv/bin/python')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.driver.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    path = output/'execution.json'
    if path.exists():
        raise FileExistsError('Inspect existing rich-control execution before any resume')
    worker = Path(__file__).with_name('isaac_rich_control_worker.py')
    source_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(a.config.read_bytes()).hexdigest()
    (output/'logs').mkdir(exist_ok=True)
    ledger = {'driver_pid': os.getpid(), 'started_utc': utc(), 'config_sha256': config_sha,
              'worker_sha256': source_sha, 'episode_ids': config['train_ids'], 'cases': [], 'completed': False}
    atomic_json(path, ledger)
    def invoke(eid, gpu):
        if hashlib.sha256(worker.read_bytes()).hexdigest() != source_sha or hashlib.sha256(a.config.read_bytes()).hexdigest() != config_sha:
            raise ValueError('Frozen rich curriculum source/configuration changed')
        cmd = [a.isaac_python, '-u', str(worker), '--config', str(a.config), '--episode', eid, '--gpu', str(gpu)]
        with (output/'logs'/(eid+'.log')).open('x') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'})
            atomic_json(output/(eid+'-worker.json'), {'pid': process.pid, 'gpu': gpu, 'command': cmd})
            code = process.wait()
        if code:
            raise RuntimeError('Isaac curriculum worker failed; data retained')
        audit = validate_case(output, eid)
        atomic_json(output/'private'/eid/'validation.json', audit)
        return {'episode_id': eid, 'gpu': gpu, 'passed': True, 'completed_utc': utc()}
    def callback(result, results):
        ledger['cases'] = results
        atomic_json(path, ledger)
        print(json.dumps(result), flush=True)
    results = run_fixed_queue(config['train_ids'], config['gpus'], invoke, on_result=callback)
    ledger.update(cases=results, completed=len(results) == len(config['train_ids']) and all(r.get('passed') for r in results), finished_utc=utc())
    atomic_json(path, ledger)
    return not ledger['completed']


if __name__ == '__main__':
    raise SystemExit(main())
