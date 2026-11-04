"""Seal a derived split-pose candidate before any evaluator process is invoked."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.gaussian_boundary_flow_v02.pose_shape_contact import predict, POLICY
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed, sha
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists(): raise FileExistsError('Derived run is immutable; choose a new directory')
    original, seal, complete, arrays = load_sealed(source)
    package = Path(original['input_package'])
    if sha(package/'manifest.json') != original['input_package_manifest_sha256']:
        raise ValueError('Changed source input manifest')
    data = load_package(package)
    if predictor_digest(data) != original['input_digest']:
        raise ValueError('Changed source inputs')
    plan = dict(original)
    plan.update(prediction_methods=['flow_pose','mixture_pose'],
                scope='Frozen derived split pose/shape plus archived rigid floor-response baseline, not coupled contact',
                derived_policy=POLICY,
                source_run=str(source.resolve()),
                parent_sha256={name:sha(source/name) for name in
                               ['plan.json','inference-sealed.json','complete.json','prediction.npz']},
                derived_source_sha256={str(path):sha(path) for path in
                                       [Path(__file__), Path('scripts/gaussian_boundary_flow_v02/pose_shape_contact.py'), Path('scripts/gaussian_boundary_flow_v02/pose_shape.py'), Path('scripts/contact_benchmark_v12/contact_rollout.py'), Path('scripts/contact_benchmark_v12/scene.py')]},
                limitations=['Pose is point-estimated; no calibrated joint pose uncertainty',
                             'Approximate rigid floor response, no inferred slip; known release only',
                             'Historical v02 parent has legacy incomplete checkpoint provenance'])
    output.mkdir(parents=True)
    write(output/'plan.json', plan)  # Freeze policy/inputs BEFORE constructing predictions.
    result = {key:arrays[key] for key in ['nodes','observation_matrix','observed_initial']}
    metadata = {}
    for family in ('flow','mixture'):
        samples = arrays[family+'_raw_samples']
        if len(samples) != complete['forecast_samples'][family]['valid']:
            raise ValueError('Stored sample count differs from source completion')
        derived, info = predict(data, samples, prefix_frames=plan['prefix_frames'], point_ids=plan['point_ids'])
        result[family+'_pose_samples'] = derived
        result[family+'_pose'] = derived.mean(0)
        metadata[family] = info
    np.savez_compressed(output/'prediction.npz', **result)
    write(output/'pose-policy-metadata.json', metadata)
    if (source/'context-diagnostic.json').exists():
        shutil.copyfile(source/'context-diagnostic.json', output/'context-diagnostic.json')
    write(output/'inference-sealed.json', dict(plan_sha256=sha(output/'plan.json'),
          forecast_target_access=False, train_end=seal['train_end'],
          derived_from_prediction_sha256=sha(source/'prediction.npz'),
          scope='Derived prediction seal; not a new posterior inference checkpoint'))
    write(output/'complete.json', dict(completed=True, held_target_access=False,
          fit_end=complete['fit_end'], frames=complete['frames'],
          forecast_samples=complete['forecast_samples'],
          prediction_sha256=sha(output/'prediction.npz'),
          policy_metadata_sha256=sha(output/'pose-policy-metadata.json'),
          no_new_sample_filtering=True, uncertainty='Pose point estimate; shape sample variability only'))
    load_sealed(output)
    print('Sealed', original['case'], str(output), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); prepare(args.source,args.output)
