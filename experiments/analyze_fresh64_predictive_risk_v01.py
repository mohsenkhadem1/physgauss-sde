"""Secondary analysis of complete, audited clean held-load predictions.

This procedure is defined after launch but before scientific results are read.
It neither changes the original experiment nor replays physics or inference.
"""
import argparse
import hashlib
import json
import os
import resource
import signal
from pathlib import Path

import numpy as np
from experiments import audit_fresh64_nonlinear_calibration_v01 as independent
from experiments import run_fresh64_nonlinear_calibration_v01 as fresh
from experiments.campaign_predictive_risk_v01 import summarize, MODELS, PHASES

ROOT = fresh.ROOT
LOCAL = ROOT/'runs/operator-dev/fresh64-predictive-risk-v01'
PRIMARY_SHA = 'ab1ecddf49342ec337401f2629d681ec88dc4f1424986872f9ed173dcbd20323'
SOURCE = 'experiments/analyze_fresh64_predictive_risk_v01.py'
FILES = (SOURCE, 'experiments/campaign_predictive_risk_v01.py',
         'tests/test_analyze_fresh64_predictive_risk_v01.py', 'tests/test_predictive_risk_v01.py')
sha, write = fresh.sha, fresh.write


def sources():
    return {name:sha(ROOT/name) for name in FILES}


def prepare():
    # No fresh observations, truths, posteriors, or object summaries are read.
    p = json.loads((fresh.LOCAL/'protocol.json').read_text())
    release = json.loads((fresh.LOCAL/'release.json').read_text())
    if sha(fresh.LOCAL/'protocol.json') != PRIMARY_SHA or release['protocol_SHA256'] != PRIMARY_SHA:
        raise ValueError('Exact immutable primary protocol required')
    for path, digest in release['audit_source_SHA256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Original independent auditor pins')
    recipe = dict(schema='fresh64-secondary-predictive-risk-v01', source_SHA256=sources(),
        primary_protocol_SHA256=PRIMARY_SHA, primary_source_SHA256=p['source_SHA256'],
        audit_source_SHA256=release['audit_source_SHA256'], objects=64, channels=96,
        models=list(MODELS), phases=list(PHASES), targets=list(fresh.study.TARGETS),
        bootstrap_seed=2026102999, bootstrap_resamples=10000,
        declared_after_launch_before_any_scientific_results_inspected=True,
        original_protocol_unchanged=True, no_new_physics_or_inference=True,
        no_pass_rule=True, no_Gaussian_coverage_claim=True,
        actual_achieved_numerical_refinement_indicators_required=True,
        diagnostic='Clean-target MSE vs mean posterior SD squared; ratio of object means, paired prefix-minus-post reductions; identical whole-object bootstrap indices for every cell; zero denominators retained as undefined.')
    path = LOCAL/'recipe.json'
    if path.exists(): raise FileExistsError('No secondary recipe refreeze')
    write(path, recipe)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def bound_archive(path, manifest):
    path = Path(path)
    if str(path) not in manifest or sha(path) != manifest[str(path)]:
        raise ValueError('Audited raw archive bytes changed')
    return np.load(path, allow_pickle=False)


def object_inputs(row, manifest, base):
    directory = Path(base)/f"object-{row['index']:02d}"
    target = []
    for label in fresh.study.TARGETS:
        with bound_archive(directory/(label+'-truth-LOCAL.npz'), manifest) as z:
            target.append(z['response_m'].copy())
    target = np.asarray(target).ravel()
    if target.shape != (96,) or not np.isfinite(target).all():
        raise ValueError('All 96 clean held-load target components required')
    mse, variance = np.empty((2,3)), np.empty((2,3))
    refinement = np.empty((2,3,2))
    for model_index, model in enumerate(MODELS):
        for phase_index, phase in enumerate(PHASES):
            record = row['posteriors'][model][phase]
            if record.get('qualified') is not True or record.get('physical_valid') is not True:
                raise ValueError('No filtering: all required posteriors must qualify')
            path = directory/(model+'-'+phase+'-posterior-LOCAL.npz')
            if sha(path) != record['raw_SHA256']: raise ValueError('Exact posterior record binding')
            with bound_archive(path, manifest) as z:
                mean, SD = z['mean'].copy(), z['SD'].copy()
            if mean.shape != (96,) or SD.shape != (96,) or not np.isfinite(mean).all() or not np.isfinite(SD).all() or np.any(SD < 0):
                raise ValueError('Finite 96-component posterior mean and nonnegative SD')
            mse[model_index, phase_index] = np.mean((mean-target)**2)
            variance[model_index, phase_index] = np.mean(SD**2)
            witnesses = record['strict_absolute_checks']['final_axis_witnesses']
            refinement[model_index, phase_index] = [max(witnesses[axis]['actual_absolute_errors_m'][key]
                for axis in ('inner','outer')) for key in ('predictive_mean','predictive_SD')]
    if not np.isfinite(refinement).all() or np.any(refinement < 0):
        raise ValueError('Finite achieved refinement indicators required')
    return mse, variance, refinement


def run(recipe_sha, audit_sha):
    if (set(os.sched_getaffinity(0)) != {17} or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))):
        raise ValueError('Secondary analysis CPU17, one thread, no GPU')
    recipe_path = LOCAL/'recipe.json'
    recipe = json.loads(recipe_path.read_text())
    if sha(recipe_path) != recipe_sha or recipe['source_SHA256'] != sources():
        raise ValueError('Frozen secondary analysis recipe/source')
    for name, digest in recipe['primary_source_SHA256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Original primary sources changed')
    for name, digest in recipe['audit_source_SHA256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Prepinned independent auditor changed')
    audit_path = independent.LOCAL/'result.json'
    audit = json.loads(audit_path.read_text())
    if (sha(audit_path) != audit_sha or audit.get('integrity_passed') is not True
            or audit.get('comparison_blocked') is not False or audit.get('objects') != 64
            or audit.get('qualified_posteriors') != 896 or audit['protocol']['sha256'] != PRIMARY_SHA):
        raise ValueError('Exact complete independent primary audit required')
    if (LOCAL/'summary.json').exists() or (LOCAL/'risk-inputs-LOCAL.npz').exists():
        raise FileExistsError('No secondary analysis overwrite')
    def deadline(*_): raise TimeoutError('300 second secondary analysis cap')
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(300)
    try:
        p, _, _, rows, manifest = independent.terminal_metadata(PRIMARY_SHA,
            audit['execution']['sha256'], audit['summary']['sha256'])
        if [row['index'] for row in rows] != list(range(64)):
            raise ValueError('All 64 ordered object clusters retained')
        inputs = [object_inputs(row, manifest, fresh.LOCAL) for row in rows]
        mse, variance, refinement = [np.stack([row[k] for row in inputs]) for k in range(3)]
        result = summarize(mse, variance, seed=recipe['bootstrap_seed'], resamples=recipe['bootstrap_resamples'])
        if not result['complete']: raise ValueError('Required complete secondary input')
        for i, cell in enumerate(result['cells']):
            worst = refinement[:, i//3, i%3].max(axis=0)
            scale = cell['RMS_posterior_SD_m']
            cell['numerical_indicators'] = dict(maximum_final_axis_mean_refinement_m=float(worst[0]),
                maximum_final_axis_SD_refinement_m=float(worst[1]),
                maximum_SD_refinement_over_RMS_posterior_SD=None if scale == 0 else float(worst[1]/scale),
                scope='Achieved successive-refinement indicators only, not rigorous error bounds, new gates, or grounds to filter objects.')
        raw = LOCAL/'risk-inputs-LOCAL.npz'
        np.savez_compressed(raw, MSE=mse, predicted_variance=variance, numerical_refinement=refinement)
        result.update(recipe_SHA256=recipe_sha, primary_protocol_SHA256=PRIMARY_SHA,
            primary_audit=dict(path=str(audit_path.relative_to(ROOT)),sha256=audit_sha),
            primary_summary=audit['summary'], raw_inputs=dict(path=str(raw.relative_to(ROOT)),sha256=sha(raw)),
            declared_after_launch_before_scientific_result_inspection=True,
            no_primary_protocol_changes=True, no_PDE_or_posterior_replay=True,
            no_parameter_or_noise_redraw=True, fresh_object_denominator=64)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
            raise MemoryError('Own 4 GiB secondary analysis cap')
        write(LOCAL/'summary.json', result)
        return dict(completed=True, summary_SHA256=sha(LOCAL/'summary.json'))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run'])
    parser.add_argument('--recipe-sha'); parser.add_argument('--audit-sha')
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else run(args.recipe_sha,args.audit_sha)))
