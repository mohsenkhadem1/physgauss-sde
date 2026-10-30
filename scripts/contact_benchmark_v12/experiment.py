"""Predict first with isolated payloads; score complete fixed cohorts afterward."""
import sys,argparse,json,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.contact_benchmark_v12.common import *
from scripts.contact_benchmark_v12.data import load,trim_prefix,predictor_payload
from scripts.contact_benchmark_v12.object_frame import adapt,select_adapter
from scripts.contact_benchmark_v12.contact_rollout import rollout
from scripts.contact_benchmark_v12.scene import load_scene,StaticScene
from scripts.external_benchmark.evaluate import score,expand_prediction,baselines
from scripts.external_benchmark.calibrated_baseline import calibrated_rigid

def base_path(data,cohort,validation=False):
    if validation:return BASE/'base/validation'/data['name']/'prediction.npz'
    if cohort=='regression':
        tag='physical-release' if data['dataset']=='SoftVTBench' else 'primary'
        return ROOT/'runs/external-benchmark-v11/fits'/tag/data['name']/'prediction.npz'
    return BASE/'base/full'/data['name']/'prediction.npz'

def predict(data,base,validation_base,validation_data,scene):
    """No evaluation dictionary is accepted by any learned/physical candidate."""
    if 'evaluation' in data or 'evaluation' in validation_data:raise ValueError('Future truth in predictor payload')
    try:
        selected,meta=select_adapter(data,base,validation_prediction=validation_base,validation_data=validation_data)
        pose,pose_meta=adapt(data,base,method='pose')
    except ValueError as error:
        # A failed base/validation solve is retained, never repaired with target
        # data or silently excluded from the cohort.
        if 'finite' not in str(error).lower() and 'failed' not in str(error).lower():raise
        selected=base.copy();pose=base.copy();meta=dict(method='none',selection_unavailable=str(error),inherited_failure=True);pose_meta=dict(unavailable=str(error))
    predictions=dict(v11_fem=base,pose_reset=pose,object_frame=selected)
    metadata=dict(object_frame=meta,pose_reset=pose_meta)
    if data['dataset']=='SoftVTBench':
        for name,collider,state in [('contact_floor_state',StaticScene(floor=data['floor']),True),('contact_scene',scene,False),('contact_scene_state',scene,True)]:
            try:predictions[name],metadata[name]=rollout(data,selected,collider,detect_detachment=state)
            except (ValueError,np.linalg.LinAlgError) as error:
                predictions[name]=selected.copy();metadata[name]=dict(contact_unavailable=str(error),inherited_failure=True)
    return predictions,metadata

def finite_metadata(value):
    if isinstance(value,dict):return {k:finite_metadata(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_metadata(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def run(spec,cohort):
    check_deadline();data=load(spec);out=BASE/'evaluation'/cohort/data['name'];out.mkdir(exist_ok=True,parents=True)
    if cohort!='development':
        freeze=json.loads((REPORT/'method-freeze.json').read_text())
        for name,digest in freeze['source_sha256'].items():
            if sha256(ROOT/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    if (out/'metrics.json').exists():return True
    source=base_path(data,cohort);validation_path=base_path(data,cohort,True)
    if not source.exists() or not validation_path.exists():return False
    # Ensure workers have committed the validation fit/prediction atomically.
    if not (validation_path.parent/'complete.json').exists():return False
    if cohort!='regression' and not (source.parent/'complete.json').exists():return False
    if data['dataset']=='SoftVTBench' and cohort=='untouched_confirmation':
        native_dir=BASE/'native/confirmation'/data['name']
        if not (native_dir/'fit.json').exists():return False
    base=np.load(source)['predicted'];validation_base=np.load(validation_path)['predicted']
    shortened=trim_prefix(data,max(3,int(data['train_end']*.7)))
    scene=load_scene(predictor_payload(data)) if data['dataset']=='SoftVTBench' else None
    provenance(out,data)
    predictions,metadata=predict(predictor_payload(data),base,validation_base,predictor_payload(shortened),scene)
    # Existing baselines access initial target only; keep them outside candidate
    # construction and record extra geometry in the geometry-matched baseline.
    baseline=baselines(data,data['train_end']);rigid=calibrated_rigid(data)
    if data['dataset']=='SoftVTBench':
        predictions['rigid_scene_state'],metadata['rigid_scene_state']=rollout(predictor_payload(data),rigid,scene)
    metadata=finite_metadata(metadata);write_json(out/'prediction-metadata.json',metadata)
    np.savez_compressed(out/'predictions.npz',**predictions)
    predictions={k:expand_prediction(data,v) for k,v in predictions.items()}
    predictions.update(baseline);predictions['prefix_calibrated_rigid']=rigid
    if data['dataset']=='PhysTwin':
        ref=ROOT/'runs/third_party/PhysTwin/reference/experiments'/data['name']/'inference.pkl'
        if ref.exists():predictions['official_phystwin']=np.asarray(pickle.load(open(ref,'rb')))@data['evaluation']['transform']
    else:
        native=(ROOT/'runs/external-benchmark-v11/native/primary' if cohort=='regression' else BASE/'native/confirmation')/data['name']/'prediction.npz'
        if native.exists():predictions['adapted_phystwin']=np.load(native)['predicted']
    start=data['train_end'];end=len(data['times']);metrics={k:score(data,p,start,end) for k,p in predictions.items()}
    segments={}
    if data['dataset']=='SoftVTBench':
        for key,mask in [('commanded_hold',data['attachment_active'].sum(1)>0),('commanded_release',(np.arange(end)>data['grasp'])&(data['gripper_command']>0))]:
            segments[key]={k:score(data,p,start,end,mask) for k,p in predictions.items()}
    write_json(out/'metrics.json',dict(name=data['name'],dataset=data['dataset'],cohort=cohort,interval=[start,end],results=metrics,segments=segments,source_base=str(source),source_sha256=sha256(source),validation_base=str(validation_path),validation_sha256=sha256(validation_path),selection=metadata['object_frame'],attachment_state=metadata.get('contact_scene_state',{}).get('attachment_state')))
    print(data['name'],{k:round(v['summary'].get('manual_track_mean_mm',v['summary'].get('coordinate_rmse_mm',-1)),3) for k,v in metrics.items()},flush=True)
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True,choices=['development','regression','untouched_confirmation']);p.add_argument('--watch',action='store_true');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);a=p.parse_args()
    protocol=json.loads((REPORT/'benchmark-protocol.json').read_text())[a.cohort]
    specs=(protocol['SoftVTBench']+[dict(case=c) for c in protocol['PhysTwin']])[a.part::a.parts]
    while True:
        done=0
        for spec in specs:
            try:done+=run(spec,a.cohort)
            except Exception as e:
                print('ERROR',spec,type(e).__name__,str(e),flush=True)
                if not a.watch:raise
        print('Cohort',a.cohort,done,'/',len(specs),flush=True)
        if done==len(specs) or not a.watch:break
        time.sleep(20)
