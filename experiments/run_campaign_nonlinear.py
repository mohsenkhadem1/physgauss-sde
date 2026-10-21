"""One fixed sequence of matched nonlinear campaign candidates on one GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from isaac_probe_pool_coordinator import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', type=Path, default=Path('configs/gaussian_campaign_nonlinear_v01.json'))
    p.add_argument('--root', type=Path, default=Path('runs/gaussian-campaign-v01/nonlinear-pilot'))
    p.add_argument('--wave', choices=['pilot', 'first8', 'first16', 'full32'], default='pilot')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root/'.driver.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    output = a.root/'execution.json'
    if output.exists():
        raise FileExistsError('Inspect existing nonlinear trial before any resume')
    cfg = json.loads(a.phase.read_text())
    ledger = {'driver_pid': os.getpid(), 'started_unix_s': time.time(), 'wave': a.wave,
              'device': a.device, 'arms': [], 'completed': False}
    atomic_json(output, ledger)
    for name in cfg['models']:
        command = [sys.executable, '-u', str(Path(__file__).with_name('train_campaign_nonlinear.py')),
            '--phase', str(a.phase), '--wave', a.wave, '--model', name,
            '--output', str(a.root/name), '--device', a.device]
        row = {'name': name, 'state': 'running', 'started_unix_s': time.time(), 'command': command}
        ledger['arms'].append(row)
        with (a.root/(name+'.log')).open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            row['pid'] = process.pid
            atomic_json(output, ledger)
            row['exit_code'] = process.wait()
        row.update(state='completed' if row['exit_code'] == 0 else 'failed', finished_unix_s=time.time())
        atomic_json(output, ledger)
        print(json.dumps(row), flush=True)
    ledger.update(completed=all(r['state'] == 'completed' for r in ledger['arms']), finished_unix_s=time.time())
    atomic_json(output, ledger)
    return not ledger['completed']


if __name__ == '__main__':
    raise SystemExit(main())
