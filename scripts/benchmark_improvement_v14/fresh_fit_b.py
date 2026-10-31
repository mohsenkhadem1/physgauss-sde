"""Fit unchanged reference mechanics on input-only prospective episodes.

This wrapper owns a new deadline and new output paths. Historical implementations
and hyperparameters remain unchanged; no evaluator is imported or read.
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import BASE, ROOT, remaining, write, sha
from scripts.prefix_ridge_switch_v13.inputs import load_payload
from scripts.contact_benchmark_v12 import base_runner
from scripts.external_benchmark import fit as fitter


BASE = BASE / 'robust'

def run(name, tag, device, field=True, continuation=False):
    remaining()
    allowed = {f'{subset}-task{task}-demo{episode}' for subset in ['object-soft','spatial-soft']
               for task in [1,2] for episode in range(40,50)}
    if name not in allowed or tag not in ['full','short']:
        raise ValueError('Case outside declared prospective cohort')
    source = BASE / 'inputs' / name / (tag + '.npz')
    data = load_payload(source)
    if 'evaluation' in data or data['name'] != name or len(data['observed']) != data['train_end']:
        raise ValueError('Invalid input-only payload')
    out = BASE / 'fresh-base' / ('current' if field else 'field_free') / tag / name
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(field=field, contact=True, nonconvex=False, materials=1)
    contract = dict(input_sha256=sha(source),metadata_sha256=sha(source.with_suffix('.json')),
                    settings=settings,maximum_iterations=100,frames=36,
                    core_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                       [ROOT/'scripts/contact_benchmark_v12/base_runner.py',
                        ROOT/'scripts/external_benchmark/fit.py',ROOT/'scripts/external_benchmark/fem.py',
                        ROOT/'scripts/external_benchmark/dynamic_fem.py']})
    if (out / 'complete.json').exists():
        saved=__import__('json').loads((out/'complete.json').read_text())
        if saved.get('contract') != contract or saved.get('prediction_sha256') != sha(out/'prediction.npz'):
            raise ValueError('Completed fit cache contract/hash mismatch')
        return
    start = time.monotonic()
    def provenance(directory, payload):
        write(directory / 'provenance.json', dict(
            input_sha256=sha(source), metadata_sha256=sha(source.with_suffix('.json')),
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                [Path(__file__), ROOT/'scripts/contact_benchmark_v12/base_runner.py',
                 ROOT/'scripts/external_benchmark/fit.py']}, device=device,
            maximum_iterations=100, frames=36, settings=settings, contract=contract,
            scope='Unchanged v12 mechanics; exported prefix and commands only'))
    fitter.check_deadline = remaining
    fitter.save_provenance = provenance
    fitter.minimize = base_runner.feasible
    base_runner.check_deadline = remaining
    if continuation:
        provenance_record = __import__('json').loads((out/'provenance.json').read_text())
        if provenance_record.get('contract') != contract:
            raise ValueError('Continuation checkpoint contract mismatch')
        record = __import__('json').loads((out/'progress.json').read_text())
        with np.load(out/'best.npz') as saved:
            theta = saved['theta'].copy()
            np.testing.assert_array_equal(saved['nodes'], data['nodes'])
            np.testing.assert_array_equal(saved['selected_frames'], np.unique(np.linspace(0,data['train_end']-1,36,dtype=int)))
        assert record['fit_end'] == data['train_end']
        np.testing.assert_array_equal(theta, record['best']['theta'])
        assert np.isfinite(theta).all() and record['best']['loss'] < 1000
        write(out/'continuation.json', dict(reason='Bounded fit timed out; unchanged valid prefix checkpoint',
              checkpoint_sha256=sha(out/'best.npz'), optimizer_converged=False))
    else:
        theta = fitter.fit(data, out, device=device, iterations=100, frames=36, **settings)
    pred, diagnostic = base_runner.rollout(data, theta, settings, device)
    temporary = out/'prediction.pending.npz'
    np.savez_compressed(temporary, predicted=pred, nodes=data['nodes'], theta=theta)
    temporary.replace(out/'prediction.npz')
    write(out/'prediction.json', diagnostic)
    write(out/'complete.json', dict(name=name, tag=tag, field=field, device=device,
        fit_end=data['train_end'], elapsed_seconds=time.monotonic()-start,
        failed_frames=len(diagnostic['failed_frames']), continuation=continuation,
        prediction_sha256=sha(out/'prediction.npz'),contract=contract))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--tag',choices=['full','short'],required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--field-free',action='store_true')
    p.add_argument('--continuation',action='store_true');a=p.parse_args()
    run(a.name,a.tag,a.device,not a.field_free,a.continuation)
