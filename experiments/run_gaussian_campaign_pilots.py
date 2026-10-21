"""Run the three frozen pilot arms sequentially on one GPU, keeping failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, default=Path('configs/gaussian_campaign_v01.json'))
    p.add_argument('--root', type=Path, default=Path('runs/gaussian-campaign-v01/pilots'))
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    handle = (a.root/'.pilot-driver.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = json.loads(a.campaign.read_text())
    ledger_path = a.root/'execution.json'
    if ledger_path.exists():
        raise FileExistsError('Inspect existing campaign pilot execution before any resume')
    ledger = {'driver_pid': os.getpid(), 'started_unix_s': time.time(), 'device': a.device,
              'arms': [], 'completed': False}
    def save():
        temp = ledger_path.with_suffix('.tmp')
        temp.write_text(json.dumps(ledger, indent=2)+'\n')
        os.replace(temp, ledger_path)
    save()
    for arm in config['initial_arms']:
        name = arm['name']
        command = [sys.executable, '-u', str(Path(__file__).with_name('train_campaign_forward.py')),
            '--campaign', str(a.campaign), '--arm', name, '--output', str(a.root/name), '--device', a.device]
        row = {'name': name, 'command': command, 'started_unix_s': time.time(), 'state': 'running'}
        ledger['arms'].append(row)
        with (a.root/(name+'.log')).open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            row['pid'] = process.pid
            save()
            row['exit_code'] = process.wait()
        row.update(state='completed' if row['exit_code'] == 0 else 'failed', finished_unix_s=time.time())
        save()
        print(json.dumps(row), flush=True)
        if row['exit_code']:
            # Preserve the failed arm and continue other predeclared arms;
            # a shared systematic preflight failure will remain visible.
            continue
    ledger['completed'] = all(r['state'] == 'completed' for r in ledger['arms'])
    ledger['finished_unix_s'] = time.time()
    save()
    return not ledger['completed']


if __name__ == '__main__':
    raise SystemExit(main())
