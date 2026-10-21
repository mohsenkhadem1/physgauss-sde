"""Posthoc static/prior context, gated on the complete sealed32 test.

Reuses preexisting C26 observation-INDEPENDENT prior means, not that object's
posterior or likelihood. No forward calls, fitting, inference or physics.
Prepare freezes sources/archives/targets before any reserved target read.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE='runs/gaussian-campaign-v01/unknown-point-oracle-v01/joint-reference-v01/'
C26_DESIGN=dict(path='runs/operator-dev/unknown-point-oracle-inputs-v01/protocol-design.json',
    sha256='f50c2de50217904b215df5e25d93b6699c19b27e48941f21233a1a1a36be9fe9')
EVALUATOR_SHA='3d82e60e5377d3faa237144fe893c58b10184a4f7757c7cfdde6d45120a91ac9'
PROGRAMS=('exciting_world_a_v01','exciting_world_b_v01')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repo-relative path required')
    p=(ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Repository escape')
    return p


def read(path):return json.loads(Path(path).read_text())
def checked(r):
    p=resolve(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('Frozen artifact changed: '+r['path'])
    return p
def record(p):return dict(path=Path(p).resolve().relative_to(ROOT).as_posix(),sha256=sha(p))
def write_new(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def prepare(output):
    old=read(checked(C26_DESIGN));gate=read(resolve(BASE+'posterior-frozen.json'))
    sources=dict(gate['source_sha256'])
    if sources.get('experiments/evaluate_campaign_point_material_reference.py')!=EVALUATOR_SHA:
        raise ValueError('Original C26 prior-generation evaluator required')
    for name in ('point_prior_context_baselines.py','render_point_prior_comparison.py','summarize_point_prior_analysis.py'):
        sources['experiments/'+name]=sha(ROOT/'experiments'/name)
    for name,digest in sources.items():checked(dict(path=name,sha256=digest))
    execution_path=resolve('runs/operator-dev/point-prior-analysis-v01/execution-protocol.json')
    execution=read(execution_path);plan=read(checked(execution['cohort_plan']))
    public=read(checked(plan['public_manifest']));oldpublic=read(checked(old['public_manifest']))
    if read(checked(plan['public_hardware']))!=read(checked(old['public_hardware'])):
        raise ValueError('Existing prior and cohort must share EXACT public H0')
    command_records={}
    for program in PROGRAMS:
        a,b=oldpublic['known_commands'][program],public['known_commands'][program]
        with np.load(checked(a),allow_pickle=False) as first,np.load(checked(b),allow_pickle=False) as second:
            names=('endpoint_times_s','output_times_s','handle_forces_world_N','handle_torques_world_Nm',
                   'nodal_forces_world_N','body_acceleration_world_m_s2')
            if set(first.files)!=set(names) or set(second.files)!=set(names) or any(not np.array_equal(first[k],second[k]) for k in names):
                raise ValueError('Existing prior/cohort controls must match every exact public array')
        command_records[program]=dict(C26=a,cohort=b)
    prior_archives={label:record(resolve(BASE+'prior-predictive-'+program+'.npz')) for label,program in zip(('A','B'),PROGRAMS)}
    design=dict(schema='point-prior-posthoc-context-v0.1',status='FROZEN_BEFORE_RESERVED_TARGET_READ',
        source_sha256=sources,C26_design=C26_DESIGN,C26_prior_generation_gate=record(resolve(BASE+'posterior-frozen.json')),
        execution=record(execution_path),cohort_plan=execution['cohort_plan'],public_hardware=plan['public_hardware'],
        known_commands=command_records,prior_archives=prior_archives,object_indices=list(range(16,48)),
        targets=dict(A=list(range(31,61)),B=list(range(61))),baselines=['static_rest','existing_C26_prior_mean'],
        comparator_policies=['no_additional_view','active'],bootstrap=dict(replicates=10000,seed=2026118000),
        prior_recipe='Exact p(mask)/Nmask prior weights, 2048 systematic samples with seed20261118 in fixed C26 evaluator; old posterior weights NEVER used',
        claim='Posthoc descriptive context, not a new preregistered primary endpoint; no tuning or new queries',
        archive_provenance='Preexisting C26 evaluator outputs first hash-bound for this context diagnostic here',
        reserved_target_or_case_values_opened=False,raw_artifacts_LOCAL=True)
    path=resolve(output);path.parent.mkdir(parents=True,exist_ok=True);write_new(path,design)
    return record(path)


def run(design_path,design_sha,evaluation_path,evaluation_sha,output):
    design=read(checked(dict(path=design_path,sha256=design_sha)))
    if (design.get('schema')!='point-prior-posthoc-context-v0.1' or design.get('status')!='FROZEN_BEFORE_RESERVED_TARGET_READ'
            or design['object_indices']!=list(range(16,48)) or design['targets']!=dict(A=list(range(31,61)),B=list(range(61)))
            or design['bootstrap']!=dict(replicates=10000,seed=2026118000)):
        raise ValueError('Exact frozen posthoc context design required')
    for name,digest in design['source_sha256'].items():checked(dict(path=name,sha256=digest))
    checked(design['C26_design']);checked(design['C26_prior_generation_gate']);checked(design['execution'])
    for rec in design['prior_archives'].values():checked(rec)
    try:
        from render_point_prior_comparison import gate
        from summarize_point_prior_analysis import paired_bootstrap
    except ImportError:
        from experiments.render_point_prior_comparison import gate
        from experiments.summarize_point_prior_analysis import paired_bootstrap
    # Complete32 release, result/source/choice/state closure BEFORE ANY target
    # or predictive arrays are loaded, using the independently reviewed gate.
    evaluation,execution,_,by_index=gate(evaluation_path,evaluation_sha)
    if (evaluation['execution_protocol_sha256']!=design['execution']['sha256']
            or execution['cohort_plan']!=design['cohort_plan']):raise ValueError('Sealed evaluation belongs to another experiment')
    destination=resolve(output)
    if destination.exists():raise ValueError('Fresh posthoc report directory required')
    rest=np.asarray(read(checked(design['public_hardware']))['rest_points_m'],dtype=float)
    prior={}
    for label,rec in design['prior_archives'].items():
        with np.load(checked(rec),allow_pickle=False) as raw:
            q=raw['mean_positions_m'].copy()
            if q.shape!=(61,len(rest),3) or not np.isfinite(q).all() or not np.array_equal(raw['times_s'],np.arange(61)/20):
                raise ValueError('Existing prior mean/common61 clock required')
        prior[label]=q
    rows=[]
    for index in range(16,48):
        value=by_index[index];row=dict(object_index=index,baselines={})
        for label in ('A','B'):
            with np.load(checked(value['capture_pairs'][label]['trajectory']),allow_pickle=False) as raw:
                truth=raw['positions_m'][design['targets'][label]].copy()
                if not np.array_equal(raw['times_s'],np.arange(61)/20):raise ValueError('Finite common clock differs')
            if truth.shape!=(len(design['targets'][label]),len(rest),3) or not np.isfinite(truth).all():raise ValueError('Complete finite target required')
            for name,mean in [('static_rest',np.broadcast_to(rest,(61,*rest.shape))),('existing_C26_prior_mean',prior[label])]:
                row['baselines'].setdefault(name,{})[label]=float(np.mean((truth-mean[design['targets'][label]])**2))
        row['policies']={name:{'A':value['states'][state]['A_after1p5s_MSE_m2'],
            'B':value['states'][state]['B_all_MSE_m2']} for name,state in (
                ('no_additional_view','no_additional_view'),('active',f"fixed_view_{value['selected_active_view']}"))}
        rows.append(row)
    result=dict(schema='point-prior-posthoc-context-result-v0.1',completed=True,objects=32,
        design=dict(path=design_path,sha256=design_sha),evaluation=dict(path=evaluation_path,sha256=evaluation_sha),
        claim=design['claim'],summaries={},paired_descriptive_intervals={},object_MSEs=rows,
        difference_sign='Context baseline MINUS sealed posterior policy; positive means baseline worse',
        uncertainty_unit='Complete independent object, not frames/rays; posthoc bootstrap intervals are descriptive',
        new_forward_inference_or_physics_calls=False,raw_artifacts_LOCAL=True)
    for name in design['baselines']:
        result['summaries'][name]={};result['paired_descriptive_intervals'][name]={}
        for label in ('A','B'):
            baseline=np.array([r['baselines'][name][label] for r in rows])
            result['summaries'][name][label]=dict(pooled_coordinate_RMSE_mm=float(1000*np.sqrt(baseline.mean())),
                mean_object_RMSE_mm=float(1000*np.sqrt(baseline).mean()))
            for policy in design['comparator_policies']:
                other=np.array([r['policies'][policy][label] for r in rows])
                result['paired_descriptive_intervals'][name].setdefault(policy,{})[label]=dict(
                    MSE_difference_m2=paired_bootstrap(baseline-other),
                    object_RMSE_difference_mm=paired_bootstrap(1000*(np.sqrt(baseline)-np.sqrt(other))))
    destination.mkdir(parents=True,exist_ok=False);write_new(destination/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('run')
    for name in ('design','design-sha256','evaluation','evaluation-sha256','output'):a.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.output)
    else:result=run(a.design,a.design_sha256,a.evaluation,a.evaluation_sha256,a.output)
    print(json.dumps(result if a.mode=='prepare' else {'completed':result['completed'],'objects':result['objects']}))
