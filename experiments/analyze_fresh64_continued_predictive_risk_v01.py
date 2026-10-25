"""Same declared secondary risk statistics on independently audited mixed lineage.

This new reader preserves the original secondary recipe and statistical helper.
Only accepted archive resolution changes. Original five-level results and their
blocked original analysis remain immutable; no physics/inference is replayed.
"""
import argparse
import importlib
import json
import os
import resource
import signal
from pathlib import Path

import numpy as np
from experiments import analyze_fresh64_predictive_risk_v01 as original

ROOT, fresh, sha, write = original.ROOT, original.fresh, original.sha, original.write
MODELS, PHASES = original.MODELS, original.PHASES
LOCAL = ROOT/'runs/operator-dev/fresh64-continued-predictive-risk-v01'
SOURCE = 'experiments/analyze_fresh64_continued_predictive_risk_v01.py'
TEST = 'tests/test_analyze_fresh64_continued_predictive_risk_v01.py'
ORIGINAL_RECIPE_SHA = '60280656bebb7d0ec30b96626754479cf1efe23aa9b3e9b8ea7d931223d38cd5'


def auditor():
    return importlib.import_module('experiments.audit_fresh64_numerical_continuation_v01')


def sources():
    return {p:sha(ROOT/p) for p in (SOURCE, TEST, original.SOURCE,
        'experiments/campaign_predictive_risk_v01.py',
        'experiments/campaign_fresh64_continuation_v01.py')}


def prepare(continuation_sha):
    # Source/config metadata only: no object records, arrays or physical values.
    independent = auditor()
    from experiments import run_fresh64_numerical_continuation_v01 as continued
    protocol_path = continued.LOCAL/'protocol.json'
    p = json.loads(protocol_path.read_text())
    old_recipe = original.LOCAL/'recipe.json'
    if sha(protocol_path) != continuation_sha or sha(old_recipe) != ORIGINAL_RECIPE_SHA:
        raise ValueError('Exact continuation and unchanged original secondary recipe')
    declared = json.loads(old_recipe.read_text())
    if (declared['bootstrap_seed'] != 2026102999 or declared['bootstrap_resamples'] != 10000
            or declared['models'] != list(MODELS) or declared['phases'] != list(PHASES)):
        raise ValueError('Unchanged original statistical targets and bootstrap')
    for name, digest in declared['source_SHA256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Original secondary sources changed')
    if (LOCAL/'recipe.json').exists(): raise FileExistsError('No recipe overwrite')
    recipe = dict(schema='fresh64-continued-secondary-risk-v01',
        continuation_protocol_SHA256=continuation_sha,
        original_recipe=dict(path=str(old_recipe.relative_to(ROOT)), sha256=ORIGINAL_RECIPE_SHA),
        source_SHA256=sources(), continuation_source_SHA256=p['source_SHA256'],
        audit_source_SHA256=independent.sources(),
        objects=64, required_accepted_posteriors=896,
        bootstrap_seed=2026102999, bootstrap_resamples=10000,
        statistics_unchanged=True, archive_resolution_only=True,
        no_new_pass_rule=True, no_filtering=True, no_PDE_or_inference=True,
        original_negative_preserved=True)
    write(LOCAL/'recipe.json',recipe)
    return dict(path=str((LOCAL/'recipe.json').relative_to(ROOT)),sha256=sha(LOCAL/'recipe.json'))


def object_inputs(row, manifest, parent_base):
    directory = Path(parent_base)/f"object-{row['index']:02d}"
    targets = []
    for label in fresh.study.TARGETS:
        with original.bound_archive(directory/(label+'-truth-LOCAL.npz'),manifest) as z:
            targets.append(z['response_m'].copy())
    target = np.asarray(targets).ravel()
    if target.shape != (96,) or not np.isfinite(target).all():
        raise ValueError('All 96 clean held-load target components required')
    mse, variance = np.empty((2,3)), np.empty((2,3))
    refinement = np.empty((2,3,2))
    for mi, model in enumerate(MODELS):
        for pi, phase in enumerate(PHASES):
            record = row['posteriors'][model][phase]
            accepted = row['posterior_lineage'][model][phase]['accepted']
            if (accepted is None or record.get('qualified') is not True
                    or record.get('physical_valid') is not True):
                raise ValueError('No filtering: all accepted posteriors must qualify')
            if accepted.get('source') not in ('original_five_levels','refined_axis_extension'):
                raise ValueError('Explicit audited original/refined lineage')
            path = ROOT/accepted['raw']['path']
            if accepted['raw']['sha256'] != record['raw_SHA256'] or sha(path) != record['raw_SHA256']:
                raise ValueError('Exact accepted posterior archive binding')
            with original.bound_archive(path,manifest) as z:
                mean, SD = z['mean'].copy(), z['SD'].copy()
            if (mean.shape != (96,) or SD.shape != (96,) or not np.isfinite(mean).all()
                    or not np.isfinite(SD).all() or np.any(SD < 0)):
                raise ValueError('Finite 96-component mean and nonnegative SD')
            mse[mi,pi] = np.mean((mean-target)**2)
            variance[mi,pi] = np.mean(SD**2)
            witnesses = record['strict_absolute_checks']['final_axis_witnesses']
            refinement[mi,pi] = [max(witnesses[a]['actual_absolute_errors_m'][k]
                for a in ('inner','outer')) for k in ('predictive_mean','predictive_SD')]
    if not np.isfinite(refinement).all() or np.any(refinement < 0):
        raise ValueError('Finite achieved refinement indicators required')
    return mse,variance,refinement


def run(recipe_sha,audit_sha):
    if (set(os.sched_getaffinity(0)) != {17} or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))):
        raise ValueError('CPU17, one thread, no GPU')
    independent = auditor()
    recipe_path = LOCAL/'recipe.json'; recipe = json.loads(recipe_path.read_text())
    if sha(recipe_path) != recipe_sha or recipe['source_SHA256'] != sources():
        raise ValueError('Frozen new reader recipe/source')
    if sha(ROOT/recipe['original_recipe']['path']) != ORIGINAL_RECIPE_SHA:
        raise ValueError('Unchanged original secondary recipe')
    for group in ('continuation_source_SHA256','audit_source_SHA256'):
        for name,digest in recipe[group].items():
            if sha(ROOT/name) != digest: raise ValueError('Frozen continued study/audit source')
    if recipe['audit_source_SHA256'] != independent.sources():
        raise ValueError('Exact independent auditor source closure')
    audit_path = independent.LOCAL/'result.json'; audit = json.loads(audit_path.read_text())
    if (sha(audit_path) != audit_sha or audit.get('integrity_passed') is not True
            or audit.get('comparison_blocked') is not False or audit.get('objects') != 64
            or audit.get('qualified_posteriors') != 896
            or audit.get('source_SHA256') != recipe['audit_source_SHA256']
            or audit.get('protocol',{}).get('sha256') != recipe['continuation_protocol_SHA256']):
        raise ValueError('Exact complete independent continued audit required')
    if (LOCAL/'summary.json').exists() or (LOCAL/'risk-inputs-LOCAL.npz').exists():
        raise FileExistsError('No secondary analysis overwrite')
    def deadline(*_): raise TimeoutError('300 second secondary analysis cap')
    previous = signal.signal(signal.SIGALRM,deadline); signal.alarm(300)
    try:
        _,_,_,_,rows,_,manifest,_ = independent.terminal_metadata(
            recipe['continuation_protocol_SHA256'],audit['execution']['sha256'],audit['summary']['sha256'])
        if [r['index'] for r in rows] != list(range(64)):
            raise ValueError('All64 ordered object clusters retained')
        inputs = [object_inputs(r,manifest,fresh.LOCAL) for r in rows]
        mse,variance,refinement = [np.stack([r[k] for r in inputs]) for k in range(3)]
        result = original.summarize(mse,variance,seed=recipe['bootstrap_seed'],resamples=recipe['bootstrap_resamples'])
        if not result['complete']: raise ValueError('Complete secondary inputs required')
        for i,cell in enumerate(result['cells']):
            worst = refinement[:,i//3,i%3].max(axis=0); scale = cell['RMS_posterior_SD_m']
            cell['numerical_indicators'] = dict(maximum_final_axis_mean_refinement_m=float(worst[0]),
                maximum_final_axis_SD_refinement_m=float(worst[1]),
                maximum_SD_refinement_over_RMS_posterior_SD=None if scale == 0 else float(worst[1]/scale),
                scope='Achieved successive-refinement indicators, not rigorous bounds, new gates, or filtering.')
        raw = LOCAL/'risk-inputs-LOCAL.npz'
        np.savez_compressed(raw,MSE=mse,predicted_variance=variance,numerical_refinement=refinement)
        result.update(recipe_SHA256=recipe_sha,continuation_protocol_SHA256=recipe['continuation_protocol_SHA256'],
            original_secondary_recipe=recipe['original_recipe'],continued_audit=dict(path=str(audit_path.relative_to(ROOT)),sha256=audit_sha),
            continued_summary=audit['summary'],raw_inputs=dict(path=str(raw.relative_to(ROOT)),sha256=sha(raw)),
            same_original_statistics=True,original_negative_preserved=True,no_PDE_or_posterior_replay=True,
            no_parameter_or_noise_redraw=True,fresh_object_denominator=64)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
            raise MemoryError('Own4GiB analysis cap')
        write(LOCAL/'summary.json',result)
        return dict(completed=True,summary_SHA256=sha(LOCAL/'summary.json'))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--continuation-sha'); parser.add_argument('--recipe-sha'); parser.add_argument('--audit-sha')
    args=parser.parse_args()
    print(json.dumps(prepare(args.continuation_sha) if args.action=='prepare' else run(args.recipe_sha,args.audit_sha)))
