"""Reuse a completed bounded MPM helper's slot for frozen extension fits."""
from pathlib import Path
import argparse
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.position_ood.study import within_deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--after-unit', required=True)
    args = parser.parse_args()
    if not args.after_unit.startswith('position-ood-v09-s10-native-mpm-single-'):
        raise ValueError('Only the bounded campaign MPM helpers are eligible')
    while True:
        within_deadline()
        output = subprocess.check_output(
            ['systemctl', '--user', 'show', args.after_unit,
             '-p', 'ActiveState', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        if state.get('ActiveState') == 'inactive' and state.get('Result') == 'success' and state.get('ExecMainStatus') == '0':
            break
        if state.get('ActiveState') == 'failed':
            raise RuntimeError(f'Preceding helper failed: {args.after_unit}')
        time.sleep(30)
    print('HANDOFF', args.after_unit, 'force-law inverse worker', args.worker, flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('force_law_queue.py')),
                    '--kind', 'inverse', '--worker', str(args.worker)], check=True)


if __name__ == '__main__':
    main()
