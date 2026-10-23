"""Evaluate only after initial, action-choice AND all postprobe seals exist.

No private target/label member is extracted before all predictions are bound.
Actual probe observations are never extracted by this evaluation program.
"""
from __future__ import annotations
import argparse,json,os,resource,signal,time,math
from pathlib import Path
import numpy as np
from experiments import run_campaign_c38_prediction_evaluation as evaluation
from experiments import campaign_c38_policy_metrics as policy_metrics

io=evaluation.io
ROOT=evaluation.ROOT
SOURCE='experiments/run_campaign_c38_full_evaluation.py'


def validate_post_seal(seal,initial_record,choices_record,ids):
    if seal.get('completed') is not True or seal.get('count')!=128 or seal.get('model_executions')!=256 or seal.get('private_target_or_latent_access') is not False or seal.get('initial_prediction_seal')!=initial_record or seal.get('choices_seal')!=choices_record:
        raise ValueError('Complete postprobe128x2 seal bound to initial and choices required')
    rows=seal.get('objects',[])
    expected={(t,f) for t in ids for f in (-2.,2.)}
    if len(rows)!=128 or {(r['object_id'],r['force_Fz_N']) for r in rows}!=expected:
        raise ValueError('All64x2 actions required; no filtering/selection')
    return rows


def run(initial_record,choices_record,post_record,observation_execution_record,output,*,root=ROOT):
    root=Path(root).resolve();dest=io.local(root,output)
    if dest.exists():raise FileExistsError('Fresh complete evaluation output only')
    if set(os.sched_getaffinity(0))!={34,35} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU34,35/noCUDA')
    initial=io.read_json(root,initial_record);choices=io.read_json(root,choices_record);post=io.read_json(root,post_record)
    ids={r['object_id'] for r in initial['objects']}
    evaluation.verify_choices(choices,initial_record,ids)
    post_rows=validate_post_seal(post,initial_record,choices_record,ids)
    bound={}
    # Complete hash barrier over EVERY postprobe file before any private label or
    # canonical-target release. Numerical negative cases remain in the seal.
    for row in post_rows:
        result=io.read_json(root,row['result'])
        if result.get('completed') is not True or result.get('object_id')!=row['object_id'] or result.get('force_Fz_N')!=row['force_Fz_N'] or set(result.get('models',{}))!=set(io.MODELS):
            raise ValueError('Exact completed two-model one-action result required')
        for model in result['models'].values():
            if model.get('execution_completed') is not True or set(model['files'])!={'posterior-arrays.npz','posterior-structure.json','numerics.json'}:raise ValueError('All postprobe model files retained')
            for f in model['files'].values():io.checked(root,f)
        bound[(row['object_id'],row['force_Fz_N'])]=result
    dest.mkdir(parents=True,exist_ok=False)
    def cap(*_):raise TimeoutError('180s full metric-only evaluation cap')
    old=signal.signal(signal.SIGALRM,cap);deadline=time.monotonic()+180;signal.alarm(180)
    try:
        initial_summary=evaluation.run(initial_record,choices_record,observation_execution_record,output+'/initial',root=root)
        # Inner bounded executor resets its alarm; restore full evaluation cap.
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('180s full evaluation cap')
        signal.alarm(math.ceil(remaining))
        execution=io.read_json(root,observation_execution_record)
        rows={name:{-2.:[],2.:[]} for name in io.MODELS};design=evaluation.obs.load_frozen_design(root)
        for record in sorted(execution['records'],key=lambda r:r['object_id']):
            with np.load(io.checked(root,record['private_future']),allow_pickle=False) as z:truth=evaluation.extract_truth_members(z)
            public=evaluation.codec.load_public_prefix(root,record['public_prefix'],design=design)
            for force in (-2.,2.):
                state=bound[(record['object_id'],force)]
                for name in io.MODELS:
                    prediction=evaluation.restore_prediction(root,state['models'][name]['files'])
                    if prediction.get('acquired_probe_force_Fz_N')!=force or prediction.get('acquired_coordinate_count')!=54 or prediction.get('fresh_prior') is not True or prediction.get('initial_posterior_weights_reused') is not False:
                        raise ValueError('Exactly prefix plus ONE probe from fresh prior required')
                    rows[name][force].append(evaluation.metrics.score_prediction(prediction,truth,public))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*1024**3:raise MemoryError('2GiB metric RSS cap')
        models={};initial_rows={}
        for name in io.MODELS:
            initial_rows[name]=json.loads((dest/'initial'/(name+'-private-metrics.json')).read_text())
            selected=choices['choices'] if name=='nonlinear_table' else choices['model_choices'][name]
            models[name]=policy_metrics.evaluate_policies(initial_rows[name],rows[name],selected)
            for force,records in rows[name].items():
                io.write_new(dest/(name+('-down' if force==-2. else '-up')+'-private-metrics.json'),io.jsonable(records))
        identities=[(r['private_configuration_index'],r['repeat_index']) for r in initial_rows['nonlinear_table']]
        linear={r['object_id']:r for r in initial_rows['linear_rest_tangent']}
        compare=None
        if all(r['admitted'] for name in io.MODELS for r in initial_rows[name]):
            compare=policy_metrics.paired_noise_summary([linear[r['object_id']]['canonical63']['coordinate_RMSE_m']-r['canonical63']['coordinate_RMSE_m'] for r in initial_rows['nonlinear_table']],identities)
            compare['positive_favors']='nonlinear_table'
        summary=dict(completed=True,initial_prediction_seal=initial_record,choices_seal=choices_record,postprobe_prediction_seal=post_record,
                     initial=initial_summary['models'],initial_nonlinear_advantage_over_linear_RMSE_m=compare,postprobe=models,
                     all_initial_choices_and_postpredictions_sealed_before_private_target_or_label_access=True,
                     actual_probe_observations_extracted_by_evaluation=False,
                     independent_population_objects=False,physical_population_calibration_established=False,
                     source_sha256={p:io.sha(root/p) for p in (SOURCE,'experiments/run_campaign_c38_prediction_evaluation.py','experiments/campaign_c38_prediction_metrics.py','experiments/campaign_c38_policy_metrics.py')})
        io.write_new(dest/'summary.json',summary);return summary
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    for n in ('initial-seal','choices-seal','post-seal','observation-execution'):
        a.add_argument('--'+n,required=True);a.add_argument('--'+n+'-sha256',required=True)
    a.add_argument('--output',required=True);v=a.parse_args()
    r=lambda n:dict(path=getattr(v,n),sha256=getattr(v,n+'_sha256'))
    s=run(r('initial_seal'),r('choices_seal'),r('post_seal'),r('observation_execution'),v.output)
    print(json.dumps(dict(completed=s['completed'],output=v.output)))
