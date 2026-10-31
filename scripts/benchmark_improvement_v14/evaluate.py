"""Separate frozen forecasting and evaluator-only benchmark scoring."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.benchmark_improvement_v14.common import BASE,ROOT,REPORT,read,write,sha,remaining
from scripts.prefix_ridge_switch_v13.inputs import load_payload,load_scene_payload
from scripts.contact_benchmark_v12.experiment import finite_metadata


def cases(dataset):
    protocol=read(REPORT/'protocol.json')
    return [c for c in protocol['historical_cases']+protocol['fresh_softvt_cases'] if c['dataset']==dataset]


def verify_freeze(dataset):
    path=REPORT/('softvt-freeze.json' if dataset=='SoftVTBench' else 'phystwin-freeze.json')
    frozen=read(path)
    if frozen['protocol_sha256']!=sha(REPORT/'protocol.json'):raise ValueError('Protocol changed')
    for file,digest in frozen['source_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen source changed: '+file)
    return sha(path)


def forecast_softvt(case):
    from scripts.benchmark_improvement_v14.provenance import snapshot_sources,validate_record
    from scripts.benchmark_improvement_v14.softvt_predict import predict as improve
    from scripts.contact_benchmark_v12.experiment import predict as original_predict
    from scripts.contact_benchmark_v12.model_selection import rigid_nodal
    from scripts.contact_benchmark_v12.contact_rollout import rollout
    frozen=verify_freeze('SoftVTBench');name=case['name'];cohort=case['cohort']
    out=BASE/'evaluation'/cohort/name
    if (out/'complete.json').exists():
        record=read(out/'complete.json')
        assert record['freeze_sha256']==frozen and record['prediction_sha256']==sha(out/'predictions.npz')
        validate_record(case,record,out)
        return True
    fresh=cohort=='prospective_softvt_confirmation'
    folder=(BASE/'inputs' if fresh else ROOT/'runs/prefix-ridge-switch-v13/inputs')/name
    data=load_payload(folder/'full.npz');scene=load_scene_payload(folder/'scene.npz')
    if fresh:
        dirs={tag:BASE/'fresh-base/current'/tag/name for tag in ['full','short']}
        if not all((d/'complete.json').exists() for d in dirs.values()):return False
    sources=snapshot_sources(case)
    if fresh:
        # Validate completed caches with the production input/settings contract.
        from scripts.benchmark_improvement_v14.fresh_fit import run as validate_fit
        for tag,directory in dirs.items():validate_fit(name,tag,read(directory/'complete.json')['device'])
        full=np.load(dirs['full']/'prediction.npz')['predicted']
        short=np.load(dirs['short']/'prediction.npz')['predicted']
        predictions,metadata=original_predict(data,full,short,load_payload(folder/'short.npz'),scene)
        rigid=rigid_nodal(data)
        predictions['prefix_calibrated_rigid']=rigid
        predictions['rigid_scene_state'],metadata['rigid_scene_state']=rollout(data,rigid,scene)
    else:
        original=ROOT/'runs/contact-benchmark-v12/evaluation'/cohort/name
        with np.load(original/'predictions.npz') as archive:
            predictions={key:archive[key].copy() for key in archive.files}
        metadata={'original':read(original/'prediction-metadata.json')}
        for directory,key in [('model-selection-evaluation','model_selection_selected'),
                              ('feasibility-evaluation','feasible_contact_scene_state')]:
            path=ROOT/'runs/contact-benchmark-v12'/directory/cohort/name/'predictions.npz'
            with np.load(path) as archive:predictions[key]=archive[key].copy()
    predictions['benchmark_v14'],metadata['benchmark_v14']=improve(
        data,predictions['contact_scene_state'],dict(scene=scene,window=8))
    verify_freeze('SoftVTBench')
    if snapshot_sources(case)!=sources:raise ValueError('Forecast inputs changed during computation')
    out.mkdir(parents=True,exist_ok=True)
    temporary=out/'predictions.pending.npz';np.savez_compressed(temporary,**predictions)
    temporary.replace(out/'predictions.npz')
    write(out/'metadata.json',finite_metadata(metadata))
    write(out/'complete.json',dict(case=case,freeze_sha256=frozen,sources=sources,
          prediction_sha256=sha(out/'predictions.npz'),metadata_sha256=sha(out/'metadata.json')))
    print('forecast',name,flush=True);return True


def score_all(dataset):
    from scripts.benchmark_improvement_v14.provenance import validate_record
    # Entire declared dataset cohort must be committed before any suffix is read.
    frozen=verify_freeze(dataset);cohort=cases(dataset)
    for case in cohort:
        out=BASE/'evaluation'/case['cohort']/case['name']
        record=read(out/'complete.json')
        if record['freeze_sha256']!=frozen or record['prediction_sha256']!=sha(out/'predictions.npz'):
            raise ValueError('Incomplete/changed forecast')
        validate_record(case,record,out)
    from scripts.contact_benchmark_v12.data import load
    from scripts.external_benchmark.evaluate import expand_prediction
    from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
    rows=[]
    for case in cohort:
        remaining();out=BASE/'evaluation'/case['cohort']/case['name'];data=load(case['spec'])
        with np.load(out/'predictions.npz') as saved:
            metrics={key:safe_score(data,expand_prediction(data,saved[key]),data['train_end'],len(data['times']))
                     for key in saved.files}
        archived=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation'/case['cohort']/case['name']/'metrics.json'
        controls={}
        if archived.exists():
            prior=read(archived)
            controls={key:value for key,value in prior['results'].items() if key not in metrics}
        record=dict(case=case,metrics=metrics,archived_controls=controls,
                    prediction_sha256=sha(out/'predictions.npz'),freeze_sha256=frozen)
        write(out/'scores.json',finite_metadata(record));rows.append(record)
        print('scored',case['name'],flush=True)
    write(REPORT/('softvt-scores.json' if dataset=='SoftVTBench' else 'phystwin-scores.json'),finite_metadata(dict(cases=rows,freeze_sha256=frozen)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['SoftVTBench','PhysTwin'],required=True)
    p.add_argument('--forecast',action='store_true');p.add_argument('--score',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args()
    if a.forecast:
        if a.dataset=='SoftVTBench':forecast=forecast_softvt
        else:
            from scripts.benchmark_improvement_v14.phystwin_evaluation import forecast
        cohort=cases(a.dataset)[a.part::a.parts]
        while True:
            remaining();done=sum(forecast(c) for c in cohort)
            print('forecast progress',a.part,done,len(cohort),flush=True)
            if done==len(cohort) or not a.watch:break
            time.sleep(20)
    if a.score:score_all(a.dataset)
