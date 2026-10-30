"""Prospective small archived-development pilot; no fitting or model selection."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,multiprocessing,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.coupled_contact_v13.campaign import remaining,write
BASE=ROOT/'runs/coupled-surface-v13/benchmark-transfer'
REPORT=ROOT/'reports/coupled-surface-v13/benchmark-transfer'
OLD=ROOT/'runs/contact-benchmark-v12/model-selection-evaluation/development'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def name(spec):return spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
def protocol():return json.loads((REPORT/'protocol.json').read_text())


def declare():
    oldprotocol=ROOT/'reports/contact-benchmark-v12/model-selection-protocol.json'
    cohort=json.loads(oldprotocol.read_text())['cohorts']['development']
    specs=cohort['SoftVTBench']+[dict(case=c) for c in cohort['PhysTwin']]
    if len(specs)!=21:raise ValueError('Expected original21 development cases')
    source=[Path(__file__),Path(__file__).with_name('pose.py')]
    source += [ROOT/'scripts'/x for x in ['contact_benchmark_v12/contact_rollout.py','contact_benchmark_v12/object_frame.py','contact_benchmark_v12/feasibility.py','contact_benchmark_v12/data.py','contact_benchmark_v12/scene.py','contact_benchmark_v12/model_selection_scoring.py','external_benchmark/data.py','external_benchmark/fem.py','external_benchmark/evaluate.py']]
    archives=[OLD/name(spec)/filename for spec in specs for filename in ['predictions.npz','metrics.json','prediction-metadata.json']]
    REPORT.mkdir(parents=True,exist_ok=True)
    write(REPORT/'protocol.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=specs,
        sources={str(p.relative_to(ROOT)):digest(p) for p in source},archived_inputs={str(p.relative_to(ROOT)):digest(p) for p in archives},
        archived_protocol_sha256=digest(oldprotocol),workers=2,
        change='Only normalized rest-tetrahedron P1 center of mass and inertia in inherited free pose rollout; same gravity, contact impulses, friction0.5,80substeps, attachment diagnosis and initial static scene.',
        frozen_selection='Use exact saved model_selection_selected trajectory; no refit, no model/adapter selection or outcome-based fallback.',
        input_policy='Predictor sees permitted prefix and commanded positions/initial geometry only; forecast targets accessed by scorer after prediction artifact saved.',
        identity_policy='Replaying original contact on original saved nodal trajectory must reproduce it exactly before variant is scored; prefix and every held frame must remain unchanged.',
        PhysTwin='Copy unchanged because archived PhysTwin method does not apply contact pose rollout.',
        status='Exploratory development-only pilot; not new generalization, coupled deformation or full PhysTwin comparison.',
        metric='SoftVT coordinateRMSEmm; PhysTwin manualTrackMeanmm. Report bothdatasets separately, all21cases, failures explicit, wins/losses/ties tolerance1e-9mm.'))
    print('Declared21cases before pilot outcomes')


def check():
    p=protocol()
    for path,sha in {**p['sources'],**p['archived_inputs']}.items():
        if digest(ROOT/path)!=sha:raise RuntimeError('Prospective source/input changed: '+path)
    if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
    return p


def run(spec):
    from scripts.contact_benchmark_v12.data import load,predictor_payload
    from scripts.contact_benchmark_v12.feasibility import inherited_elements
    from scripts.contact_benchmark_v12.scene import load_scene
    from scripts.contact_benchmark_v12.contact_rollout import rollout as original
    from scripts.coupled_contact_v13.benchmark_transfer.pose import rollout
    from scripts.contact_benchmark_v12.model_selection_scoring import safe_score
    from scripts.external_benchmark.evaluate import expand_prediction
    start=time.monotonic();case=name(spec);out=BASE/case;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    try:
        if remaining()<=0:raise RuntimeError('Original campaign deadline reached')
        data=load(spec);payload=predictor_payload(data)
        with np.load(OLD/case/'predictions.npz') as archive:old=archive['model_selection_selected'].copy()
        if data['dataset']=='SoftVTBench':
            scene=load_scene(payload);replay,oldmeta=original(payload,old,scene)
            if not np.array_equal(replay,old,equal_nan=True):raise RuntimeError('Original frozen pose replay differs; do not score variant')
            prediction,metadata=rollout(payload,old,scene,inherited_elements(payload))
            changed=np.zeros(len(old),bool);changed[[r['frame'] for r in metadata['diagnostics']]]=True
            if not np.array_equal(prediction[~changed],old[~changed],equal_nan=True):raise RuntimeError('Held/prefix states changed')
        else:prediction=old.copy();metadata=dict(policy='Unchanged PhysTwin; archived scene contact rollout unused');changed=np.zeros(len(old),bool)
        # Prediction is persisted before the evaluation targets are scored.
        np.savez_compressed(out/'prediction.npz',predicted=prediction,previous=old)
        write(out/'prediction-metadata.json',metadata)
        metrics={key:safe_score(data,expand_prediction(data,values),data['train_end'],len(data['times'])) for key,values in [('previous',old),('volume_pose',prediction)]}
        key='coordinate_rmse_mm' if data['dataset']=='SoftVTBench' else 'manual_track_mean_mm'
        before=metrics['previous']['summary'].get(key);after=metrics['volume_pose']['summary'].get(key)
        complete=metrics['volume_pose']['summary']['failed_frames']==0 and before is not None and after is not None
        row=dict(case=case,dataset=data['dataset'],complete=complete,metric=key,previous=before,volume_pose=after,
                 gain_mm=before-after if before is not None and after is not None else None,
                 integrated_free_frames=int(changed.sum()),bitwise_unchanged=bool(np.array_equal(prediction,old,equal_nan=True)),
                 results=metrics,seconds=time.monotonic()-start,prediction_sha256=digest(out/'prediction.npz'),
                 archive_sha256=digest(OLD/case/'predictions.npz'),protocol_sha256=digest(REPORT/'protocol.json'))
    except Exception as error:row=dict(case=case,dataset='PhysTwin' if 'case' in spec else 'SoftVTBench',complete=False,error=repr(error),seconds=time.monotonic()-start)
    write(out/'metrics.json',row);return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args()
    if args.declare:declare();return
    p=check()
    if not (REPORT/'protocol-audit.txt').exists():raise RuntimeError('Independent prospective protocol audit required before pilot')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(run,p['cases']))
    check();summary={}
    for dataset in ['SoftVTBench','PhysTwin']:
        allrows=[r for r in rows if r['dataset']==dataset];valid=[r for r in allrows if r['complete']];g=np.array([r['gain_mm'] for r in valid])
        summary[dataset]=dict(expected=len(allrows),complete=len(valid),mean_previous_mm=float(np.mean([r['previous'] for r in valid])) if valid else None,
            mean_volume_pose_mm=float(np.mean([r['volume_pose'] for r in valid])) if valid else None,
            wins=int((g>1e-9).sum()),losses=int((g< -1e-9).sum()),ties=int((np.abs(g)<=1e-9).sum()),
            unchanged_cases=sum(r.get('bitwise_unchanged',False) for r in allrows),means_conditional_on_complete=len(valid)!=len(allrows))
    write(REPORT/'results.json',dict(scope=p['status'],summary=summary,cases=rows,protocol_sha256=digest(REPORT/'protocol.json')))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
