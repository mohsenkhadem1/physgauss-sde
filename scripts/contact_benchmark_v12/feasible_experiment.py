"""Separate, predeclared topology safeguard; frozen v12 predictions stay intact."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import ROOT,BASE,REPORT,check_deadline,sha256,write_json
from scripts.contact_benchmark_v12.data import load,predictor_payload
from scripts.contact_benchmark_v12.experiment import run as original_run,finite_metadata
from scripts.contact_benchmark_v12.feasibility import repair,inherited_elements,jacobians,_temporal_summary
from scripts.contact_benchmark_v12.scene import load_scene
from scripts.contact_benchmark_v12.contact_rollout import rollout
from scripts.external_benchmark.evaluate import score,expand_prediction

COHORTS=['development','regression','untouched_confirmation','fresh_confirmation']
FREEZE=REPORT/'feasibility-freeze.json'

def specifications(cohort):
    protocol=json.loads((REPORT/('feasibility-protocol.json' if cohort=='fresh_confirmation' else 'benchmark-protocol.json')).read_text())[cohort]
    return protocol['SoftVTBench']+[dict(case=c) for c in protocol['PhysTwin']]

def topology_audit(data,trajectory,elements):
    rows=[]
    for t,points in enumerate(trajectory):
        if not np.isfinite(points).all():rows.append(dict(frame=t,failed=True));continue
        J=jacobians(data['nodes'],points,elements)
        rows.append(dict(frame=t,failed=False,min_J=float(J.min()),inverted_elements=int((J<=0).sum()),below_solver_margin=int((J<=.03).sum())))
    finite=[r for r in rows if not r['failed']]
    return dict(elements=len(elements),frames=len(rows),failed_frames=sum(r['failed'] for r in rows),
        min_J=min((r['min_J'] for r in finite),default=None),
        frames_with_inversions=sum(r['inverted_elements']>0 for r in finite),
        frames_below_solver_margin=sum(r['below_solver_margin']>0 for r in finite),per_frame=rows)

def predict(payload,base,candidate,scene):
    if 'evaluation' in payload:raise ValueError('Future targets cannot enter feasible prediction')
    elements=inherited_elements(payload)
    corrected,meta=repair(payload,base,candidate,elements=elements)
    predictions={'feasible_object_frame':corrected};metadata={'guard':meta}
    if payload['dataset']=='SoftVTBench':
        try:
            predictions['feasible_contact_scene_state'],metadata['contact']=rollout(payload,corrected,scene,detect_detachment=True,friction=.5,substeps=80)
        except (ValueError,np.linalg.LinAlgError) as error:
            predictions['feasible_contact_scene_state']=corrected.copy()
            metadata['contact']=dict(contact_unavailable=str(error),inherited_failure=True)
    metadata['topology']={key:topology_audit(payload,value,elements) for key,value in predictions.items()}
    return predictions,metadata

def check_freeze():
    frozen=json.loads(FREEZE.read_text())
    assert frozen['protocol_sha256']==sha256(REPORT/'feasibility-protocol.json')
    for path,digest in frozen['source_sha256'].items():assert sha256(ROOT/path)==digest,path

def run(spec,cohort):
    check_deadline();check_freeze()
    name=spec.get('case') or f"{spec['subset']}-task{spec['task']}-demo{spec['episode']}"
    output=BASE/'feasibility-evaluation'/cohort/name
    if (output/'metrics.json').exists():return True
    original=BASE/'evaluation'/cohort/name
    if cohort=='fresh_confirmation':
        if 'case' not in spec and not (BASE/'native/confirmation'/name/'fit.json').exists():return False
        if not original_run(spec,cohort):return False
    if not (original/'metrics.json').exists():return False
    data=load(spec);payload=predictor_payload(data)
    saved=np.load(original/'predictions.npz');base=saved['v11_fem'];candidate=saved['object_frame']
    scene=load_scene(payload) if data['dataset']=='SoftVTBench' else None
    predictions,metadata=predict(payload,base,candidate,scene)
    primary='feasible_contact_scene_state' if data['dataset']=='SoftVTBench' else 'feasible_object_frame'
    old_primary='contact_scene_state' if data['dataset']=='SoftVTBench' else 'object_frame'
    metadata['final_temporal_change_vs_uncorrected']=_temporal_summary(predictions[primary]-saved[old_primary],data['times'])
    original_metric=json.loads((original/'metrics.json').read_text())
    # All predictions, including fair baseline, are generated before evaluator scores.
    supplement_source=original/'native-scene-supplement.json'
    supplement=None
    if data['dataset']=='SoftVTBench' and cohort!='development':
        if cohort=='fresh_confirmation':
            source=BASE/'native/confirmation'/name/'prediction.npz';native=np.load(source)
            np.testing.assert_array_equal(native['nodes'],data['nodes'])
            predictions['adapted_phystwin_scene_state'],metadata['native_scene']=rollout(payload,native['predicted'],scene,detect_detachment=True,friction=.5,substeps=80)
            metadata['native_source_sha256']=sha256(source)
        else:
            if not supplement_source.exists():return False
            supplement=json.loads(supplement_source.read_text())
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions.npz',**predictions)
    write_json(output/'prediction-metadata.json',finite_metadata(metadata))
    start=data['train_end'];end=len(data['times'])
    results=original_metric['results'].copy()
    results.update({key:score(data,expand_prediction(data,value),start,end) for key,value in predictions.items()})
    if supplement:results[supplement['method']]=supplement['result']
    write_json(output/'metrics.json',dict(name=name,dataset=data['dataset'],cohort=cohort,primary_method=primary,interval=[start,end],
        results=results,original_metrics_sha256=sha256(original/'metrics.json'),original_prediction_sha256=sha256(original/'predictions.npz'),
        guard_metadata_sha256=sha256(output/'prediction-metadata.json'),prediction_sha256=sha256(output/'predictions.npz'),
        protocol_sha256=sha256(REPORT/'feasibility-protocol.json'),freeze_sha256=sha256(FREEZE),
        topology=metadata['topology'][primary],temporal_change=metadata['final_temporal_change_vs_uncorrected']))
    print('Feasible',cohort,name,'complete',flush=True);return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=COHORTS,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    specs=specifications(a.cohort)[a.part::a.parts]
    while True:
        done=0
        for spec in specs:done+=run(spec,a.cohort)
        print('Feasibility cohort',a.cohort,done,'/',len(specs),flush=True)
        if done==len(specs) or not a.watch:break
        check_deadline();time.sleep(20)

if __name__=='__main__':main()
