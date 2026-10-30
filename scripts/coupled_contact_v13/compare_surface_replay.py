"""Compare a successful independent replay against its archived declared job."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def compare(replay):
    replay = Path(replay).resolve()
    request = json.loads((replay / 'replay-request.json').read_text())
    execution = json.loads((replay / 'replay-execution.json').read_text())
    selector = Path(request['job']['path'])
    if selector.is_absolute() or '..' in selector.parts:
        raise ValueError('Unsafe job selector')
    archived = ROOT / 'runs/coupled-surface-v13' / selector
    fresh = replay / 'outputs' / selector
    old = json.loads((archived / 'complete.json').read_text())
    new = json.loads((fresh / 'complete.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result_path = replay / 'replay-result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    provenance = {
        'execution_request_hash': execution.get('request_sha256') == digest(replay / 'replay-request.json'),
        'execution_result_hash': result_path.exists() and execution.get('replay_result_sha256') == digest(result_path),
        'request_job': request.get('job') == old.get('job') == new.get('job'),
        'request_settings': request.get('simulation_settings') == old.get('simulation_settings') == new.get('simulation_settings'),
        'request_protocol': request.get('protocol_sha256') == old.get('protocol_sha256') == new.get('protocol_sha256'),
        'result_status': result.get('status') == new and result.get('success') is True,
        'result_job': result.get('job') == request.get('job') and result.get('simulation_settings') == request.get('simulation_settings'),
    }
    required_artifacts = {str((fresh / n).relative_to(replay)) for n in ('complete.json', 'trajectory.npz', 'diagnostics.json')}
    inventory = result.get('artifact_sha256', {})
    provenance['result_artifacts'] = set(inventory) == required_artifacts
    for rel, expected in inventory.items():
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe replay artifact path')
        provenance['artifact:' + rel] = (replay / relative).is_file() and digest(replay / relative) == expected
    metadata = {k: old[k] == new[k] for k in
                ('job', 'simulation_settings', 'protocol_sha256', 'success', 'diagnostics')}
    arrays = {}
    with np.load(archived / 'trajectory.npz', allow_pickle=False) as a, np.load(
            fresh / 'trajectory.npz', allow_pickle=False) as b:
        required = {'frames', 'velocities', 'centers', 'times', 'reactions', 'rest', 'tets', 'tool_positions', 'active'}
        keys_match = set(a.files) == set(b.files) and required <= set(a.files)
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                arrays[key] = {'equal': False, 'reason': 'missing array'}
                continue
            x, y = a[key], b[key]
            same_shape = x.shape == y.shape
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all()) if (
                np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)) else True
            equal = same_shape and x.dtype == y.dtype and np.array_equal(x, y)
            delta = None
            if same_shape and finite and x.size and np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                delta = float(np.max(np.abs(x.astype(float) - y.astype(float))))
            arrays[key] = {'equal': bool(equal), 'finite': finite,
                           'shape_match': same_shape, 'max_absolute_difference': delta}
    inputs = [replay / 'replay-request.json', replay / 'replay-execution.json',
              archived / 'complete.json', fresh / 'complete.json',
              archived / 'trajectory.npz', fresh / 'trajectory.npz']
    if result_path.exists(): inputs.append(result_path)
    return {
        'scope': 'Exact same-machine single-job reproduction; not a full-study or cross-platform guarantee',
        'job': request['job'], 'metadata_equal': metadata, 'array_keys_equal': keys_match,
        'arrays': arrays, 'provenance_checks': provenance,
        'passed': bool(execution['returncode'] == 0 and not execution['timed_out']
                       and old['success'] and new['success'] and all(metadata.values())
                       and all(provenance.values()) and keys_match and all(v['equal'] and v.get('finite', False) for v in arrays.values())),
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.replay)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'passed': result['passed'], 'arrays': len(result['arrays'])}))
    raise SystemExit(0 if result['passed'] else 1)
