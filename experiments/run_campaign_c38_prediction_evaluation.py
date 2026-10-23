"""PRIVATE metric executor; no training, choices, probing or observation draws.

Prepare source before outcomes, then invoke only after the complete initial
prediction and policy-choice seals exist. Opening the private archive extracts
ONLY canonical targets and labels, never actual candidate-probe outcomes.
"""
from __future__ import annotations
import argparse, json, os, resource, signal
from pathlib import Path
import numpy as np
from experiments import campaign_c38_prediction_metrics as metrics
from experiments import run_campaign_c38_initial_inference as io
from experiments import run_campaign_c38_controlled_observations as codec
from experiments import campaign_c38_controlled_observations as obs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'experiments/run_campaign_c38_prediction_evaluation.py'
SCALAR_STRING = {'object_id','support_category'}
SCALAR_INT = {'private_configuration_index','repeat_index'}
SCALAR_FLOAT = {'young_Pa','canonical_target_Fz_N'}


def extract_truth_members(archive):
    """Selective NPZ access: do not materialize future measurement members."""
    out = {}
    for key in sorted(metrics.TRUTH_KEYS):
        a = np.asarray(archive[key])
        if key in SCALAR_STRING:
            if a.shape != () or a.dtype.kind != 'U': raise ValueError('Scalar Unicode labels')
            out[key] = str(a.item())
        elif key in SCALAR_INT:
            if a.shape != () or a.dtype != np.int64: raise ValueError('Scalar int64 configuration/repeat')
            out[key] = int(a.item())
        elif key in SCALAR_FLOAT:
            if a.shape != () or a.dtype != np.float64: raise ValueError('Scalar float64 known target/load or material')
            out[key] = float(a.item())
        else: out[key] = a.copy()
    metrics.truth_snapshot(out)
    return out


def restore_prediction(root, files):
    if set(files) != {'posterior-arrays.npz','posterior-structure.json','numerics.json'}:
        raise ValueError('Exact saved posterior bundle required')
    for r in files.values(): io.checked(root,r)
    tree = io.read_json(root,files['posterior-structure.json'])
    with np.load(io.checked(root,files['posterior-arrays.npz']),allow_pickle=False) as z:
        used = set()
        def unpack(v):
            if isinstance(v,dict) and set(v)=={'array','shape','dtype'}:
                key=v['array'];a=z[key]
                if list(a.shape)!=v['shape'] or str(a.dtype)!=v['dtype']: raise ValueError('Saved array descriptor mismatch')
                used.add(key);return a.copy()
            if isinstance(v,dict): return {k:unpack(x) for k,x in v.items()}
            if isinstance(v,list): return [unpack(x) for x in v]
            return v
        result=unpack(tree)
        if used!=set(z.files): raise ValueError('Every saved array must be bound by posterior structure')
    return result


def verify_choices(seal, initial_record, ids):
    # Explicit v01 contract agreed before any actual probe outcome access.
    if seal.get('completed') is not True or seal.get('count')!=64 or seal.get('private_probe_outcomes_access') is not False or seal.get('initial_prediction_seal')!=initial_record:
        raise ValueError('Complete pre-outcome policy choices bound to initial seal required')
    rows=seal.get('choices',[])
    if len(rows)!=64 or {r['object_id'] for r in rows}!=ids:
        raise ValueError('All64 policy choices required before private evaluation')
    for row in rows:
        choices=row['choices']
        if set(choices)!={'information','target_risk','fixed_down','fixed_up','random'} or choices['fixed_down']!=-2. or choices['fixed_up']!=2. or any(v not in (-2.,2.) for v in choices.values()):
            raise ValueError('Five equal one-probe policy choices required')
    return rows


def run(initial_record, choices_record, observation_execution_record, output, *, root=ROOT):
    root=Path(root).resolve();dest=io.local(root,output)
    if dest.exists(): raise FileExistsError('Exclusive evaluation output; preserve failures')
    if set(os.sched_getaffinity(0))!={34,35} or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU34,35 and no CUDA required')
    def cap(*_): raise TimeoutError('120s metric-only evaluation cap')
    old=signal.signal(signal.SIGALRM,cap);signal.alarm(120)
    try:
        initial=io.read_json(root,initial_record); choices=io.read_json(root,choices_record)
        if initial.get('completed') is not True or initial.get('count')!=64 or initial.get('model_executions')!=128 or initial.get('private_truth_access') is not False:
            raise ValueError('Complete64x2 pre-truth prediction seal required')
        ids={r['object_id'] for r in initial['objects']}
        if len(ids)!=64: raise ValueError('Unique64 prediction identities')
        choice_rows=verify_choices(choices,initial_record,ids)
        for r in choice_rows: io.checked(root,r['result'])
        # Verify ALL posterior files before first private target/label access.
        bound={}
        for row in initial['objects']:
            state=io.read_json(root,row['result'])
            if state.get('completed') is not True or state.get('object_id')!=row['object_id'] or set(state['models'])!=set(io.MODELS):
                raise ValueError('Exact two-model saved object result')
            for model in state['models'].values():
                for r in model['files'].values(): io.checked(root,r)
            bound[row['object_id']]=state
        # Only now release private execution metadata and selected truth members.
        execution=io.read_json(root,observation_execution_record)
        if execution.get('completed') is not True or len(execution['records'])!=64 or {r['object_id'] for r in execution['records']}!=ids:
            raise ValueError('Complete same64 observation production proof')
        for r in execution['records']:
            io.checked(root,r['private_future']);io.checked(root,r['public_prefix'])
        dest.mkdir(parents=True,exist_ok=False);model_rows={m:[] for m in io.MODELS};design=obs.load_frozen_design(root)
        for r in sorted(execution['records'],key=lambda x:x['object_id']):
            with np.load(io.checked(root,r['private_future']),allow_pickle=False) as z: truth=extract_truth_members(z)
            public=codec.load_public_prefix(root,r['public_prefix'],design=design)
            for name in io.MODELS:
                prediction=restore_prediction(root,bound[r['object_id']]['models'][name]['files'])
                model_rows[name].append(metrics.score_prediction(prediction,truth,public))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*1024**3: raise MemoryError('2GiB evaluation RSS cap')
        for name,rows in model_rows.items(): io.write_new(dest/(name+'-private-metrics.json'),io.jsonable(rows))
        summary=dict(completed=True,initial_prediction_seal=initial_record,choices_seal=choices_record,
                     observation_execution=observation_execution_record,models={k:metrics.aggregate(v) for k,v in model_rows.items()},
                     source_sha256={p:io.sha(root/p) for p in (SOURCE,'experiments/campaign_c38_prediction_metrics.py')},
                     initial_and_choices_sealed_before_private_target_or_label_access=True,actual_probe_observations_extracted=False,
                     physical_population_calibration_established=False,raw_private_metrics_local=True)
        io.write_new(dest/'summary.json',summary)
        return summary
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    for n in ('initial-seal','choices-seal','observation-execution'):
        a.add_argument('--'+n,required=True);a.add_argument('--'+n+'-sha256',required=True)
    a.add_argument('--output',required=True);v=a.parse_args()
    r=lambda n:dict(path=getattr(v,n),sha256=getattr(v,n+'_sha256'))
    result=run(r('initial_seal'),r('choices_seal'),r('observation_execution'),v.output)
    print(json.dumps(dict(completed=result['completed'],models=result['models'])))
