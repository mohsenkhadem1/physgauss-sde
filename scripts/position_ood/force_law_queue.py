"""Resumable execution for the separately sealed force-law extension."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.force_law_study import BASE, ROOT, SPLIT, DATA_SEAL, verify, seal_data
from scripts.position_ood.study import within_deadline
from scripts.position_ood.seal import verify as verify_main
from scripts.phystwin.sequential_reference import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['reference', 'inverse', 'native'], required=True)
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    verify()
    settings = verify_main()['settings']
    directory = BASE/'force-law-jobs'
    directory.mkdir(exist_ok=True)
    tasks = []
    for i in range(6):
        if args.kind == 'reference':
            tasks.append((f'reference-{i}', ['--task', 'reference', '--object', str(i)], BASE/f'observations/{SPLIT}-{i}.npz'))
        elif args.kind == 'inverse':
            tasks.append((f'neural-{i}', ['--task', 'neural', '--object', str(i)], BASE/f'neural-predictions/{SPLIT}-{i}-selected-all.npz'))
            for method in ['extrapolated', 'extrapolated_material_frame']:
                tasks.append((f'{method}-{i}', ['--task', 'inverse', '--object', str(i), '--method', method], BASE/f'inverse/{SPLIT}-{i}/{method}-n3-p3.json'))
        else:
            for method in ['mpm', 'phystwin']:
                tag = settings[method]['tag']
                dest = BASE/f'{tag}/{SPLIT}-{i}/summary.json' if method == 'mpm' else BASE/f'phystwin/{SPLIT}-{i}/{tag}/summary.json'
                tasks.append((f'{method}-{i}', ['--task', method, '--object', str(i)], dest))
    while True:
        within_deadline()
        if args.kind != 'reference' and not DATA_SEAL.exists():
            time.sleep(30)
            continue
        pending = False
        progress = False
        for name, arguments, output in tasks:
            status = directory/(name+'.json')
            if output.exists():
                continue
            if status.exists() and json.loads(status.read_text()).get('state') == 'failed':
                continue
            pending = True
            with (directory/(name+'.lock')).open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if output.exists():
                    continue
                verify()
                start = time.time()
                write(status, dict(state='running', worker=args.worker, start=start, output=str(output)))
                command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/position_ood/force_law_study.py'), *arguments]
                print('FORCE_LAW_START', name, flush=True)
                env = os.environ.copy()
                if args.kind == 'reference':
                    env['POSITION_REFERENCE_DEVICE'] = 'cpu'
                with (directory/(name+'.log')).open('a') as log:
                    result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                success = result.returncode == 0 and output.exists()
                write(status, dict(state='complete' if success else 'failed', worker=args.worker,
                                   elapsed_seconds=time.time()-start, returncode=result.returncode, output=str(output)))
                print('FORCE_LAW_DONE', name, success, flush=True)
                progress = True
                break
        if not pending:
            if args.kind == 'reference' and all(output.exists() for _, _, output in tasks):
                seal_data()
            write(directory/f'worker-{args.kind}-{args.worker}-done.json', dict(time=time.time()))
            return
        if not progress:
            time.sleep(30)


if __name__ == '__main__':
    main()
