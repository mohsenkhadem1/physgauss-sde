"""Metric-only C39 development extension; sealed predictions before SIDE scoring.

C38 targets have already been examined. The C39 design was frozen before that
analysis; this is NOT a new sealed population test. Original prefix/DOWN metric
rows are hash-bound before SIDE observations and reused exactly, never refitted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import numpy as np
from experiments import run_campaign_c38_prediction_evaluation as evaluation
from experiments import campaign_c39_policy_metrics as metrics

io = evaluation.io
ROOT = evaluation.ROOT
SOURCE = 'experiments/run_campaign_c39_full_evaluation.py'
DESIGN = dict(path='runs/operator-dev/c39-controlled-extension-design-v01/design.json',
              sha256='3555d4ad8235efa72b1c606e626cbab262a1fb7c1729b4f79f6a9fd669f0e321')
C38_SUMMARY = dict(path='runs/gaussian-campaign-v01/c38-full-evaluation-v01/summary.json',
                   sha256='c37ce28a038293b7eda67e1e1f9bf542a33edcfa3bf1962f42ca562dbe14e82f')
FILES = {'posterior-arrays.npz', 'posterior-structure.json', 'numerics.json'}


def source_records(root):
    names = [SOURCE, 'experiments/campaign_c39_policy_metrics.py',
             'experiments/campaign_c38_policy_metrics.py',
             'experiments/campaign_c38_prediction_metrics.py',
             'experiments/run_campaign_c38_prediction_evaluation.py']
    return {name: io.sha(root / name) for name in names}


def freeze_reference(protocol, *, root=ROOT):
    """Freeze old derived rows and metric definitions before any SIDE outcome."""
    root = Path(root).resolve()
    io.checked(root, DESIGN)
    old = io.read_json(root, C38_SUMMARY)
    if old.get('completed') is not True:
        raise ValueError('Completed old controlled reference required')
    for name, digest in old['source_sha256'].items():
        if io.sha(root / name) != digest:
            raise ValueError('Old metric source changed')
    rows = {}
    for name in io.MODELS:
        prefix = f'runs/gaussian-campaign-v01/c38-full-evaluation-v01/initial/{name}-private-metrics.json'
        down = f'runs/gaussian-campaign-v01/c38-full-evaluation-v01/{name}-down-private-metrics.json'
        rows[name] = dict(initial=io.record(root, root / prefix), down=io.record(root, root / down))
    state = dict(schema='C39-fixed-metric-reference-v01', design=DESIGN,
                 C38_completed_summary=C38_SUMMARY, source_sha256=source_records(root),
                 initial_prediction_seal=old['initial_prediction_seal'],
                 C38_choices_seal=old['choices_seal'], C38_postseal=old['postprobe_prediction_seal'],
                 old_metric_rows=rows, C38_private_targets_previously_analyzed=True,
                 new_SIDE_observations_available_at_freeze=False,
                 conditional_noise_bootstrap_replicates=5000,
                 conditional_noise_bootstrap_seed=2026102304,
                 raw_and_numerical_quantile_coverage_reported_separately=True,
                 no_truth_dependent_interval_adjustment=True)
    io.write_new(io.local(root, protocol), state)
    return io.sha(io.local(root, protocol))


def check_bundle(root, result):
    if result.get('completed') is not True or set(result.get('models', {})) != set(io.MODELS):
        raise ValueError('Completed two-model result required')
    for model in result['models'].values():
        if model.get('execution_completed') is not True or set(model.get('files', {})) != FILES:
            raise ValueError('Every saved model bundle, including negatives, required')
        for record in model['files'].values():
            io.checked(root, record)


def verify_choices(root, seal, initial_record, ids):
    if (seal.get('schema') != 'C39-typed-static-policy-choices-seal-v01' or
            seal.get('completed') is not True or seal.get('count') != 64 or
            seal.get('model_executions') != 128 or seal.get('extension_design') != DESIGN or
            seal.get('private_SIDE_outcomes_access') is not False or
            seal.get('acquired_DOWN_observations_or_posteriors_access') is not False or
            seal.get('private_target_or_latent_access') is not False or
            seal.get('all_choices_sealed_before_SIDE_noise') is not True or
            seal.get('initial_prediction_seal') != initial_record):
        raise ValueError('All64 pre-outcome choices bound to old initial seal')
    if set(seal.get('model_choices', {})) != set(io.MODELS):
        raise ValueError('Both model policy choices required')
    for model, rows in seal['model_choices'].items():
        if len(rows) != 64 or {r['object_id'] for r in rows} != ids:
            raise ValueError('Complete matched choice identities')
        for r in rows:
            c = r['choices']
            if (set(c) != set(metrics.POLICIES) or c['fixed_down'] != metrics.DOWN or
                    c['fixed_side'] != metrics.SIDE or any(a not in (metrics.DOWN, metrics.SIDE) for a in c.values())):
                raise ValueError('Known typed one-action choices only')
    rows = seal.get('choices', [])
    if len(rows) != 64 or {r['object_id'] for r in rows} != ids:
        raise ValueError('All64 saved choice results required')
    for r in rows:
        io.checked(root, r['result'])


def barrier(root, reference, choices_record, post_record):
    """Hash all reused and new predictions before new private metric access."""
    initial_record = reference['initial_prediction_seal']
    initial = io.read_json(root, initial_record)
    if initial.get('completed') is not True or initial.get('count') != 64 or initial.get('model_executions') != 128:
        raise ValueError('Old complete initial64x2 seal')
    ids = {r['object_id'] for r in initial['objects']}
    if len(ids) != 64:
        raise ValueError('64 unique opaque IDs')
    for r in initial['objects']:
        check_bundle(root, io.read_json(root, r['result']))
    oldpost = io.read_json(root, reference['C38_postseal'])
    if (oldpost.get('completed') is not True or oldpost.get('initial_prediction_seal') != initial_record or
            oldpost.get('choices_seal') != reference['C38_choices_seal']):
        raise ValueError('Exact original postprobe provenance')
    down = [r for r in oldpost['objects'] if r['force_Fz_N'] == -2.]
    if len(down) != 64 or {r['object_id'] for r in down} != ids:
        raise ValueError('All64 original DOWN posterior results')
    for r in down:
        state = io.read_json(root, r['result'])
        if state.get('object_id') != r['object_id'] or state.get('force_Fz_N') != -2.:
            raise ValueError('Original DOWN identity and force')
        check_bundle(root, state)
    choices = io.read_json(root, choices_record)
    verify_choices(root, choices, initial_record, ids)
    post = io.read_json(root, post_record)
    if (post.get('completed') is not True or post.get('count') != 64 or post.get('model_executions') != 128 or
            post.get('private_target_or_latent_access') is not False or
            post.get('initial_prediction_seal') != initial_record or post.get('choices_seal') != choices_record):
        raise ValueError('Complete sealed SIDE64x2 before private scoring')
    if len(post.get('objects', [])) != 64 or {r['object_id'] for r in post['objects']} != ids:
        raise ValueError('Every SIDE posterior retained')
    bound = {}
    for r in post['objects']:
        state = io.read_json(root, r['result'])
        if (r.get('action_id') != metrics.SIDE or state.get('action_id') != metrics.SIDE or
                state.get('force_world_N') != [0., 2., 0.] or state.get('object_id') != r['object_id']):
            raise ValueError('Typed SIDE identity/world force, no Fz relabel')
        check_bundle(root, state)
        bound[r['object_id']] = state
    for rows in reference['old_metric_rows'].values():
        for r in rows.values():
            io.checked(root, r)
    return ids, choices, bound


def run(reference_record, choices_record, post_record, observation_execution_record, output, *, root=ROOT):
    root = Path(root).resolve()
    dest = io.local(root, output)
    if dest.exists():
        raise FileExistsError('Exclusive metric output; preserve all failures')
    if set(os.sched_getaffinity(0)) != {34, 35} or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU34/35 and no CUDA')
    def cap(*_):
        raise TimeoutError('180s metric-only cap')
    previous = signal.signal(signal.SIGALRM, cap)
    signal.alarm(180)
    try:
        reference = io.read_json(root, reference_record)
        if (reference.get('schema') != 'C39-fixed-metric-reference-v01' or reference.get('design') != DESIGN or
                reference.get('C38_completed_summary') != C38_SUMMARY or reference.get('source_sha256') != source_records(root)):
            raise ValueError('Frozen source/reference unchanged')
        io.checked(root, DESIGN)
        io.checked(root, C38_SUMMARY)
        ids, choices, bound = barrier(root, reference, choices_record, post_record)
        # Only after all prediction files are verified: release old labels/target
        # members, NEVER candidate-probe observations. C38 already saw these labels.
        execution = io.read_json(root, observation_execution_record)
        if execution.get('completed') is not True or len(execution['records']) != 64 or {r['object_id'] for r in execution['records']} != ids:
            raise ValueError('Exact original64 observations and labels')
        for r in execution['records']:
            io.checked(root, r['private_future'])
            io.checked(root, r['public_prefix'])
        dest.mkdir(parents=True)
        rows = {name: [] for name in io.MODELS}
        design = evaluation.obs.load_frozen_design(root)
        for r in sorted(execution['records'], key=lambda r: r['object_id']):
            with np.load(io.checked(root, r['private_future']), allow_pickle=False) as z:
                truth = evaluation.extract_truth_members(z)
            public = evaluation.codec.load_public_prefix(root, r['public_prefix'], design=design)
            for name in io.MODELS:
                prediction = evaluation.restore_prediction(root, bound[r['object_id']]['models'][name]['files'])
                if (prediction.get('acquired_probe_action_id') != metrics.SIDE or
                        not np.array_equal(prediction.get('acquired_probe_force_world_N'), [0., 2., 0.]) or
                        prediction.get('acquired_coordinate_count') != 54 or prediction.get('fresh_prior') is not True or
                        prediction.get('initial_posterior_weights_reused') is not False):
                    raise ValueError('SIDE plus prefix from fresh original prior required')
                rows[name].append(evaluation.metrics.score_prediction(prediction, truth, public))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2 * 1024**3:
                raise MemoryError('2GiB metric cap')
        models = {}
        for name in io.MODELS:
            old = reference['old_metric_rows'][name]
            before = io.read_json(root, old['initial'])
            down = io.read_json(root, old['down'])
            models[name] = metrics.evaluate_policies(before, {metrics.DOWN: down, metrics.SIDE: rows[name]}, choices['model_choices'][name])
            io.write_new(dest / (name + '-SIDE-private-metrics.json'), io.jsonable(rows[name]))
        summary = dict(completed=True, design=DESIGN, frozen_reference=reference_record, choices_seal=choices_record,
                       SIDE_prediction_seal=post_record, models=models, raw_private_metrics_local=True,
                       C38_targets_previously_analyzed=True, C39_design_frozen_before_C38_target_analysis=True,
                       all_SIDE_predictions_sealed_before_SIDE_dependent_scoring=True,
                       actual_probe_observations_extracted_by_evaluation=False,
                       independent_population_objects=False, physical_population_calibration_established=False,
                       source_sha256=source_records(root))
        io.write_new(dest / 'summary.json', summary)
        return summary
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference-sha256')
    for name in ('choices-seal', 'post-seal', 'observation-execution'):
        parser.add_argument('--' + name)
        parser.add_argument('--' + name + '-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.mode == 'freeze':
        print(freeze_reference(args.reference))
    else:
        record = lambda n: dict(path=getattr(args, n), sha256=getattr(args, n + '_sha256'))
        result = run(record('reference'), record('choices_seal'), record('post_seal'), record('observation_execution'), args.output)
        print(json.dumps(dict(completed=result['completed'], output=args.output)))
