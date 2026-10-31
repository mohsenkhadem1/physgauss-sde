"""Additive robust SoftVT comparison; all220 forecasts precede new40 outcome access."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import BASE as PARENT,ROOT,REPORT,read,write,sha,remaining
from scripts.benchmark_improvement_v14.provenance_b import snapshot_sources,validate_record
from scripts.prefix_ridge_switch_v13.inputs import load_payload,load_scene_payload
from scripts.contact_benchmark_v12.experiment import finite_metadata
BASE=PARENT/'robust'


def cases():
    p=read(REPORT/'protocol-b.json');return p['regression_cases']+p['fresh_softvt_cases']


def freeze():
    path=REPORT/'softvt-freeze-b.json';record=read(path)
    if record['protocol_sha256']!=sha(REPORT/'protocol-b.json'):raise ValueError('B protocol changed')
    for file,digest in record['source_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('B frozen source changed: '+file)
    return sha(path)


def forecast(case):
    from scripts.benchmark_improvement_v14.softvt_robust import predict
    from scripts.benchmark_improvement_v14.softvt_predict import predict as phase_a
    from scripts.contact_benchmark_v12.experiment import predict as original_predict
    from scripts.contact_benchmark_v12.model_selection import rigid_nodal
    from scripts.contact_benchmark_v12.contact_rollout import rollout
    frozen=freeze();name=case['name'];cohort=case['cohort'];out=BASE/'evaluation'/cohort/name
    if (out/'complete.json').exists():
        record=read(out/'complete.json');validate_record(case,record,out)
        if record['freeze_sha256']!=frozen:raise ValueError('B cache freeze changed')
        return True
    fresh=cohort=='prospective_softvt_phase_b'
    if fresh:
        dirs={tag:BASE/'fresh-base/current'/tag/name for tag in ['full','short']}
        if not all((d/'complete.json').exists() for d in dirs.values()):return False
        folder=BASE/'inputs'/name
    else:
        folder=(PARENT/'inputs' if cohort=='prospective_softvt_confirmation' else ROOT/'runs/prefix-ridge-switch-v13/inputs')/name
    sources=snapshot_sources(case);data=load_payload(folder/'full.npz');scene=load_scene_payload(folder/'scene.npz')
    if fresh:
        full=np.load(dirs['full']/'prediction.npz')['predicted'];short=np.load(dirs['short']/'prediction.npz')['predicted']
        predictions,metadata=original_predict(data,full,short,load_payload(folder/'short.npz'),scene)
        rigid=rigid_nodal(data);predictions['prefix_calibrated_rigid']=rigid
        predictions['rigid_scene_state'],metadata['rigid_scene_state']=rollout(data,rigid,scene)
        predictions['phase_a_prediction'],metadata['phase_a_prediction']=phase_a(data,predictions['contact_scene_state'],dict(scene=scene,window=8))
    else:
        previous=PARENT/'evaluation'/cohort/name
        with np.load(previous/'predictions.npz') as saved:predictions={key:saved[key].copy() for key in saved.files}
        predictions['phase_a_prediction']=predictions.pop('benchmark_v14')
        metadata=dict(phase_a_metadata_sha256=sha(previous/'metadata.json'))
    predictions['benchmark_v14'],metadata['benchmark_v14']=predict(data,predictions['contact_scene_state'],dict(scene=scene,window=8))
    if sources!=snapshot_sources(case):raise ValueError('B inputs changed during prediction')
    freeze();out.mkdir(parents=True,exist_ok=True)
    pending=out/'predictions.pending.npz';np.savez_compressed(pending,**predictions);pending.replace(out/'predictions.npz')
    write(out/'metadata.json',finite_metadata(metadata));write(out/'complete.json',dict(case=case,sources=sources,freeze_sha256=frozen,
       prediction_sha256=sha(out/'predictions.npz'),metadata_sha256=sha(out/'metadata.json')))
    print('B forecast',name,flush=True);return True


def matched_frames(proposed,baseline):
    a={r['frame']:r for r in proposed['frames'] if not r.get('failed')};b={r['frame']:r for r in baseline['frames'] if not r.get('failed')}
    common=sorted(set(a)&set(b));keys=['coordinate_rmse_mm','aligned_shape_rmse_mm']
    return dict(frames=common,count=len(common),proposal={k:float(np.mean([a[t][k] for t in common])) for k in keys} if common else {},
                baseline={k:float(np.mean([b[t][k] for t in common])) for k in keys} if common else {},
                scope='Matched available frames only; missing baseline frames stay explicitly unscored')


def score_all():
    frozen=freeze();cohort=cases()
    for case in cohort:
        out=BASE/'evaluation'/case['cohort']/case['name'];record=read(out/'complete.json')
        validate_record(case,record,out)
        if record['freeze_sha256']!=frozen:raise ValueError('B forecast freeze mismatch')
    from scripts.contact_benchmark_v12.data import load
    from scripts.external_benchmark.evaluate import expand_prediction
    from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
    rows=[]
    for case in cohort:
        remaining();out=BASE/'evaluation'/case['cohort']/case['name'];data=load(case['spec'])
        with np.load(out/'predictions.npz') as saved:
            metrics={key:safe_score(data,expand_prediction(data,saved[key]),data['train_end'],len(data['times'])) for key in saved.files}
        controls={};prior=PARENT/'evaluation'/case['cohort']/case['name']/'scores.json'
        if prior.exists():controls=read(prior).get('archived_controls',{})
        pairs={k:matched_frames(metrics['benchmark_v14'],metrics[k]) for k in ['contact_scene_state','rigid_scene_state','phase_a_prediction'] if k in metrics}
        row=dict(case=case,metrics=metrics,archived_controls=controls,matched_frame_comparisons=pairs,
                 prediction_sha256=sha(out/'predictions.npz'),freeze_sha256=frozen)
        write(out/'scores.json',finite_metadata(row));rows.append(row);print('B scored',case['name'],flush=True)
    write(REPORT/'softvt-scores-b.json',finite_metadata(dict(cases=rows,freeze_sha256=frozen)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forecast',action='store_true');p.add_argument('--score',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    if a.forecast:
        cohort=cases()[a.part::a.parts]
        while True:
            remaining();done=sum(forecast(c) for c in cohort);print('B progress',a.part,done,len(cohort),flush=True)
            if done==len(cohort) or not a.watch:break
            time.sleep(20)
    if a.score:score_all()
