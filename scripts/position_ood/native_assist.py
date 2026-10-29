"""Execute unchanged declared baseline jobs alongside inverse workers.

Shares the original per-job lock and status files with campaign_jobs.py. Only
execution concurrency changes; all commands come from the frozen manifest.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE, ROOT, within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.campaign_jobs import jobs
from scripts.phystwin.sequential_reference import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--script', choices=['fit_mpm.py', 'phystwin.py', 'predict_selected.py', 'uncertainty.py'])
    parser.add_argument('--max-jobs', type=int)
    args = parser.parse_args()
    tasks = [j for j in jobs(verify()['settings'])
             if (j['script'] == args.script if args.script else j['environment'] == 'native')]
    completed = 0
    directory = BASE / 'jobs'
    while True:
        within_deadline()
        pending = False
        progress = False
        for job in tasks:
            output = Path(job['result'])
            status = directory / (job['name'] + '.json')
            if output.exists():
                continue
            if status.exists() and json.loads(status.read_text()).get('state') in ('failed', 'blocked'):
                continue
            pending = True
            if not all(Path(p).exists() for p in job['deps']):
                continue
            with (directory / (job['name'] + '.lock')).open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if output.exists():
                    continue
                verify()
                python = ROOT / ('runs/phystwin-env/bin/python'
                                 if job['environment'] == 'native' else '.venv/bin/python')
                command = [str(python),
                           str(ROOT / 'scripts/position_ood' / job['script']), *job['args']]
                execution_kind = ('cpu' if job['environment'] != 'native'
                                  and os.environ.get('POSITION_INVERSE_DEVICE') == 'cpu' else 'gpu')
                start = time.time()
                write(status, dict(state='running', worker=args.worker, kind=execution_kind,
                                   job=job, start=start, executor='native_assist'))
                print('JOB_START', job['name'], flush=True)
                with (directory / (job['name'] + '.log')).open('a') as log:
                    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                            stdout=log, stderr=subprocess.STDOUT)
                success = result.returncode == 0 and output.exists()
                write(status, dict(state='complete' if success else 'failed', worker=args.worker,
                                   kind=execution_kind, job=job, returncode=result.returncode,
                                   elapsed_seconds=time.time()-start, executor='native_assist'))
                print('JOB_DONE', job['name'], success, flush=True)
                completed += 1
                if args.max_jobs is not None and completed >= args.max_jobs:
                    return
                progress = True
                break
        if not pending:
            return
        if not progress:
            time.sleep(20)


if __name__ == '__main__':
    main()
