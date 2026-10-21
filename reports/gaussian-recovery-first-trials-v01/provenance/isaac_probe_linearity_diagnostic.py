"""TRAIN diagnostic: unchanged verified Isaac worker, explicit zero-force menu.

Only the public menu is replaced: retain+x,+y,+zpulses, add zero force, retain
the diagonal and+xreset replay. Physics, initial state, hardware, capture and
all acceptance thresholds are inherited verbatim from the frozen v0.3 worker.
Wrapper provenance is explicit at pool root, additional to base-worker records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import isaac_calibrated_probes_v03 as base

BASE_SHA256 = '116d4717bff0c7780925bd4e6d85f5b2f1615d33f4993e356a8284199829a821'
ORIGINAL_MENU = base.calibrated_menu


def diagnostic_menu(*args, **kwargs):
    menu = ORIGINAL_MENU(*args, **kwargs)
    menu['candidates'] = menu['candidates'][:3]+[{**menu['candidates'][0],
        'action_id': 'zero_force', 'kind': 'zero_force_diagnostic', 'peak_force_N': [0., 0., 0.]}]
    menu['protocol'] = 'train-zero-offset-and-linearity-diagnostic-v0.1'
    menu['qualification'] = 'Physics/sensors unchanged; TRAIN-only diagnostic menu includes zero force; never an active-policy comparison'
    return menu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode', required=True)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    case = next(c for c in json.loads((a.dataset/'plan.json').read_text())['episodes'] if c['episode_id'] == a.episode)
    if case['split'] != 'train' or a.episode not in {'t0000', 't0001', 't0002', 't0003'}:
        raise ValueError('Four declared TRAIN diagnostic cases only')
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Frozen verified simulator worker changed')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {'protocol': 'train-zero-offset-and-linearity-diagnostic-v0.1',
        'base_worker_sha256': BASE_SHA256,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'change': 'Menu only:3axis pulses, zero force, diagonal query and+xreset replay',
        'physics_changes': False, 'sensor_tolerance_changes': False,
        'test_or_validation_cases_allowed': False}
    path = a.output/'diagnostic-protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Diagnostic protocol/source changed')
    else:
        with path.open('x') as f:
            json.dump(protocol, f, indent=2)
            f.write('\n')
    sources = a.output/'diagnostic-provenance'
    sources.mkdir(exist_ok=True)
    target = sources/Path(__file__).name
    if target.exists() and target.read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Diagnostic wrapper source changed')
    shutil.copy2(__file__, target)
    base.calibrated_menu = diagnostic_menu
    return base.run_worker(a)


if __name__ == '__main__':
    raise SystemExit(main())
