"""Use one idle worker slot for one unchanged calibration fine fit.

No prefix preparation or probe choice is performed here. Shared assist and
inverse-output locks coordinate with the original workers. The helper exits
after one fit so it cannot permanently expand GPU concurrency.
"""
from pathlib import Path
import fcntl
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import BASE, within_deadline
from scripts.position_ood.seal import verify
from scripts.position_ood.run_inverse import run


def main():
    within_deadline()
    settings = verify()['settings']
    directory = BASE / 'active-jobs'
    split = 'transfer_calibration'
    method = 'extrapolated'
    for i in range(12):
        choices = directory / f'{split}-{i}-choices.json'
        if not choices.exists():
            continue
        actions = sorted(set(json.loads(choices.read_text())['actions'].values()))
        for action in actions:
            assert action in [3, 4, 5, 6, 8]
            dest = BASE / f'inverse/{split}-{i}'
            tag = f'p4-0-1-2-{action}'
            output = dest / f'{method}-n3-{tag}.json'
            if (output.exists() or not (dest / f'joint-n2-{tag}.json').exists()
                    or not (dest / f'{method}-n3-p3.json').exists()):
                continue
            with (directory / f'assist-{split}-{i}-{method}-{action}.lock').open('a') as claim:
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                with output.with_suffix('.lock').open('a') as peek:
                    try:
                        fcntl.flock(peek, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    fcntl.flock(peek, fcntl.LOCK_UN)
                if output.exists():
                    continue
                within_deadline()
                print('ASSIST_ONCE_START', split, i, method, action, flush=True)
                # run() independently locks and rechecks the output, including
                # any race with an original worker after the nonblocking peek.
                run(split, i, method, 3, settings['iterations'], (0, 1, 2, action))
                print('ASSIST_ONCE_DONE', split, i, method, action, flush=True)
                return
    print('ASSIST_ONCE_NO_ELIGIBLE_FIT', flush=True)


if __name__ == '__main__':
    main()
