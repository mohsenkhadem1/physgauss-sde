"""Root operator entry point: independent checkpoint audit after external reap.

No physics execution or release of the remaining groups is performed here.
The precommit predates inspection of checkpoint scientific results.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments import audit_fixture_balanced_nh_checkpoint_v01 as audit
from experiments.supervise_surrogate_sbc_continuation import supervise

OP = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = json.loads((OP / 'root-procedure-precommit.json').read_text())
    for path, digest in p['source_SHA256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Changed root-precommitted audit/watchdog source: ' + path)
    if set(os.sched_getaffinity(0)) != {18}:
        raise ValueError('Root audit watchdog requires CPU18')
    protocol = audit.stage.LOCAL / 'protocol.json'
    if sha(protocol) != p['protocol_SHA256']:
        raise ValueError('Exact frozen physical protocol')
    frozen = audit.stage.validate(p['protocol_SHA256'])
    # Full hash/closure barrier before launching the array-reading child.
    audit.frozen.phase_metadata('checkpoint', frozen, p['protocol_SHA256'])
    command = [
        'taskset', '-c', '16', sys.executable, '-u', '-m',
        'experiments.audit_fixture_balanced_nh_checkpoint_v01',
        '--protocol-sha', p['protocol_SHA256'],
        '--source-sha', p['source_SHA256'][audit.SOURCE],
        '--test-sha', p['source_SHA256'][audit.TEST],
    ]
    result = supervise(command, OP / 'audit-stdout-LOCAL.log',
                       OP / 'supervisor-receipt.json',
                       wall_s=3600, RSS_bytes=4 * 1024**3)
    print(json.dumps(result), flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    sys.exit(main())
