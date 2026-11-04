"""Prefix-only nearest simulator-library lookup. No evaluator imports or fitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import predictor_digest
from scripts.gaussian_boundary_flow_v02.object_backend_geometry import GeometryAwareGaussianBoundaryBackend
from scripts.gaussian_boundary_flow_v02.object_backend_wide import WideGaussianTableBoundaryBackend


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def retrieve(data,plan,features,latents,k=1):
    frames=np.asarray(plan['prefix_frames']);ids=np.asarray(plan['point_ids'])
    if frames.dtype.kind not in 'iu' or frames.ndim!=1 or not len(frames) or frames[0]!=0 or np.any(np.diff(frames)<=0) or frames[-1]>=data['train_end']:
        raise ValueError('Invalid prefix frames')
    if ids.dtype.kind not in 'iu' or ids.ndim!=1 or not len(ids) or np.any(ids<0) or np.any(ids>=len(data['observed_initial'])) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid observation point IDs')
    observed=data['observed'][frames][:,ids]
    mask=np.repeat(data['observed_mask'][frames][:,ids,None],3,axis=-1).reshape(-1)
    query=(observed-observed[:1]).reshape(-1)[mask]
    if not len(query) or not np.isfinite(query).all():raise ValueError('No finite query features')
    if features.ndim!=2 or features.shape[1]!=len(query) or latents.ndim!=2 or len(latents)!=len(features) or not np.isfinite(features).all() or not np.isfinite(latents).all():raise ValueError('Invalid bank arrays')
    if int(mask.sum())!=plan['coordinate_mask_count']:raise ValueError('Changed feature mask')
    if not 1<=k<=len(features):raise ValueError('Insufficient bank worlds')
    errors=np.mean((features-query[None])**2,axis=1)
    ids=np.argsort(errors,kind='stable')[:k]
    return ids,errors,query


def load_source(run):
    run=Path(run)
    complete=json.loads((run/'complete.json').read_text());seal=json.loads((run/'inference-sealed.json').read_text());plan=json.loads((run/'plan.json').read_text())
    if not complete.get('completed') or complete.get('held_target_access')is not False or seal.get('forecast_target_access')is not False:raise ValueError('Source campaign not complete and target-free')
    if sha(run/'plan.json')!=seal['plan_sha256'] or sha(run/'bank.npz')!=seal['bank_sha256'] or sha(run/'prediction.npz')!=complete['prediction_sha256']:raise ValueError('Changed source seal')
    provenance={'inference_sha256','models_sha256','transform_sha256'}
    if provenance.intersection(seal):
        if not provenance.issubset(seal) or seal.get('schema')!=1:raise ValueError('Incomplete source provenance seal')
        files={'inference-sealed.npz':seal['inference_sha256'],'transform.pt':seal['transform_sha256']}
        if not isinstance(seal['models_sha256'],dict)or not seal['models_sha256']:raise ValueError('Invalid source model manifest')
        for name,digest in seal['models_sha256'].items():
            if not isinstance(name,str)or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',name):raise ValueError('Invalid source model name')
            files[name+'.pt']=digest
        for name,digest in files.items():
            if not(run/name).is_file()or sha(run/name)!=digest:raise ValueError('Changed source inference provenance: '+name)
    package=Path(plan['input_package'])
    if sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:raise ValueError('Changed input manifest')
    data=load_package(package)
    if predictor_digest(data)!=plan['input_digest'] or data['name']!=plan['case'] or data['train_end']!=seal['train_end']:raise ValueError('Changed source inputs')
    with np.load(run/'bank.npz',allow_pickle=False)as f:features=f['features'].copy();latents=f['z'].copy()
    return plan,data,features,latents


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--top8',action='store_true');p.add_argument('--device',default='cuda:2');a=p.parse_args()
    if a.device not in ('cpu','cuda:0','cuda:1','cuda:2'):raise ValueError('Unauthorized device')
    torch.set_num_threads(2);source=Path(a.source);out=Path(a.output)
    old,data,features,latents=load_source(source);selected,errors,query=retrieve(data,old,features,latents,8 if a.top8 else 1)
    geometry=old['args'].get('geometry','filtered')
    if geometry not in ('filtered','convex'):raise ValueError('Unknown source geometry contract')
    backend=GeometryAwareGaussianBoundaryBackend if geometry=='filtered'else WideGaussianTableBoundaryBackend
    b=backend(data,a.device)
    if b.names!=old['parameters'] or [b.lower.tolist(),b.upper.tolist()]!=old['bounds']:raise ValueError('Changed backend parameter contract')
    # The source records these files; reject changed equations before reusing bank worlds.
    for name,digest in old['source_sha256'].items():
        if 'object_backend_' in name and sha(name)!=digest:raise ValueError('Changed source backend: '+name)
    methods=['nearest_raw','nearest_tracking']+(['top8_raw','top8_tracking']if a.top8 else [])
    plan={**old,'prediction_methods':methods,'scope':'Classical nearest-library retrieval using prefix feature MSE only; geometry-specific bank, no neural inference, optimization, likelihood or calibrated posterior claim',
          'retrieval':{'source':str(source.resolve()),'source_plan_sha256':sha(source/'plan.json'),'source_bank_sha256':sha(source/'bank.npz'),'selected_ids':selected.tolist(),'selected_prefix_mse_m2':errors[selected].tolist(),'ensemble':'Uniform top8, descriptive retrieval variability only'if a.top8 else None},
          'retrieval_source_sha256':sha(Path(__file__))}
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    np.savez_compressed(out/'inference-sealed.npz',selected_ids=selected,selected_latents=latents[selected],prefix_mse_m2=errors,query=query)
    write(out/'inference-sealed.json',{'plan_sha256':sha(out/'plan.json'),'bank_sha256':sha(source/'bank.npz'),'selection_sha256':sha(out/'inference-sealed.npz'),'forecast_target_access':False,'train_end':data['train_end']})
    worlds=[];diagnostics=[];start=time.monotonic()
    for rank,index in enumerate(selected):
        try:
            world,_=b.rollout(latents[index],np.arange(len(data['times'])))
            frames=np.asarray(old['prefix_frames']);ids=np.asarray(old['point_ids'])
            mask=np.repeat(data['observed_mask'][frames][:,ids,None],3,axis=-1).reshape(-1)
            observed=b.observation_features(world[frames],frames,ids)
            replay=(observed-observed[:1]).reshape(-1)[mask]
            if not np.allclose(replay,features[index],rtol=1e-7,atol=1e-8):raise ValueError('Retrieved world prefix differs from stored bank feature; backend replay mismatch')
            worlds.append(world);diagnostics.append({'bank_index':int(index),'prefix_replay_max_abs_m':float(np.max(np.abs(replay-features[index]))),**b.last_path_diagnostics})
        except (RuntimeError,ValueError)as e:
            failure={'rank':rank,'bank_index':int(index),'error':str(e),'selection_changed':False}
            write(out/f'failure-{rank}.json',failure)
            worlds.append(np.full((len(data['times']),len(data['nodes']),3),np.nan));diagnostics.append(failure)
    worlds=np.stack(worlds);tracking=data['nodes'][None,None]+worlds-worlds[:,:1]
    predictions={'nearest_raw':worlds[0],'nearest_tracking':tracking[0]}
    if a.top8:predictions.update(top8_raw=worlds.mean(0),top8_tracking=tracking.mean(0),retrieval_worlds=worlds)
    # No *_samples channels: selected neighbors are NOT posterior samples.
    np.savez_compressed(out/'prediction.npz',**predictions,**{key:data[key]for key in ['nodes','observation_matrix','observed_initial']})
    write(out/'complete.json',{'completed':True,'fit_end':data['train_end'],'frames':len(data['times']),'held_target_access':False,'prediction_sha256':sha(out/'prediction.npz'),'elapsed_s':time.monotonic()-start,'retrieval_diagnostics':diagnostics,'failure_policy':'Failed frozen selections retain NaN trajectories; ensemble is invalid if any selected member fails; no substitution or omission','uncertainty':'No posterior or calibration claim; top8 variation, if stored, is descriptive retrieval variation'})
    print(json.dumps({'output':str(out),'selected_ids':selected.tolist(),'prefix_rmse_mm':(1000*np.sqrt(errors[selected])).tolist()},indent=2))

if __name__=='__main__':main()
