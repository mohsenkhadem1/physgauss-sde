"""Geometry evaluation kept separate from fitting; explicit development/test modes."""
import argparse,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import phystwin,softvt,interpolate_matrix,rigid_map

def expand_prediction(data,pred):
    initial=data.get('support_geometry',data['evaluation']['points'][0]);A=interpolate_matrix(data['nodes'],initial)
    return initial[None]+np.einsum('pn,tnc->tpc',A,pred-data['nodes'][None])

def features(data):
    target=data['controller_targets'];nodes=data['nodes']
    displacement=target-nodes[None,None]
    # Mean and affine linear moments of known positional attachment targets.
    X=nodes-nodes.mean(0);norm=max(np.max(np.ptp(nodes,axis=0)),1e-6)
    means=displacement.mean(2)/norm
    moments=np.einsum('tgni,nj->tgij',displacement/norm,X/norm)/len(nodes)
    return np.c_[means.reshape(len(target),-1),moments.reshape(len(target),-1),data['attachment_active']]

def baselines(data,fit_end):
    target=data['evaluation']['points'];initial=data.get('support_geometry',target[0]);T=len(target)
    result={'persistence':np.broadcast_to(initial,(T,len(initial),3)).copy()}
    rigid=[]
    for t in range(T):
        active=data['attachment_active'][t]
        if active.sum()==0:
            if data['dataset']=='SoftVTBench' and t>data['grasp'] and len(rigid)>1:
                dt=float(data['times'][t]-data['times'][t-1])
                velocity=(rigid[-1].mean(0)-rigid[-2].mean(0))/dt
                step=velocity*dt+np.array([0.,0.,-9.81*dt*dt])
                points=rigid[-1]+step
                points[:,2]+=max(0.,data['floor']-points[:,2].min())
                rigid.append(points)
            else:rigid.append(initial.copy())
            continue
        desired=np.average(data['controller_targets'][t],axis=0,weights=active)
        R,b=rigid_map(data['nodes'],desired);rigid.append(initial@R+b)
    result['control_rigid']=np.asarray(rigid)
    # Ridge regression fits identical sparse observed tracks; unseen nodes interpolate.
    features_all=features(data);F=features_all[:fit_end]
    mean=F.mean(0);std=F.std(0);std[std<1e-5]=1
    F=(F-mean)/std;F_all=(features_all-mean)/std
    X=np.c_[np.ones(fit_end),F];XX=np.c_[np.ones(T),F_all]
    obs=data['observed'][:fit_end]-data['observed_initial'][None]
    visible=data['observed_mask'][:fit_end]
    prediction=np.zeros((T,obs.shape[1],3))
    for i in range(obs.shape[1]):
        use=visible[:,i];design=X[use];y=obs[use,i]
        coeff=np.linalg.solve(design.T@design+np.eye(X.shape[1])*.1,design.T@y)
        prediction[:,i]=XX@coeff
    # RBF interpolation from observed initial locations, not target shape.
    src=data['observed_initial'];d,ids=cKDTree(src).query(initial,k=min(6,len(src)))
    w=np.exp(-((d/np.maximum(d[:,-1,None],1e-9))**2)*3)/np.maximum(d,1e-6)
    w/=w.sum(1,keepdims=True)
    result['position_ridge']=initial[None]+np.einsum('pk,tpkc->tpc',w,prediction[:,ids])
    return result

def kabsch_residual(reference,x):
    R,b=rigid_map(reference,x)
    return x-(reference@R+b)

def score(data,pred,start,end,selected=None):
    target=data['evaluation']['points'];mask=data['evaluation']['mask'];rows=[]
    for t in range(start,min(end,len(pred))):
        if selected is not None and not selected[t]:continue
        valid=mask[t]&np.isfinite(target[t]).all(1)
        if not np.isfinite(pred[t]).all():rows.append(dict(frame=t,failed=True));continue
        corresponding=pred[t,:len(target[t])]
        err=corresponding[valid]-target[t,valid]
        # Remove best rigid transform between prediction and target, exposing shape error.
        shape=kabsch_residual(corresponding[valid],target[t,valid])
        rows.append(dict(frame=t,mean_l2_mm=float(np.linalg.norm(err,axis=1).mean()*1000),
                         coordinate_rmse_mm=float(np.sqrt(np.mean(err**2))*1000),
                         aligned_shape_rmse_mm=float(np.sqrt(np.mean(shape**2))*1000),
                         observed_to_predicted_chamfer_l1_mm=float(cKDTree(pred[t,:data['evaluation'].get('surface_count',pred.shape[1])]).query(target[t,valid],p=1)[0].mean()*1000)))
    valid=[r for r in rows if not r.get('failed')]
    summary={key:float(np.mean([r[key] for r in valid])) for key in ['mean_l2_mm','coordinate_rmse_mm','aligned_shape_rmse_mm','observed_to_predicted_chamfer_l1_mm']} if valid else {}
    summary.update(total_frames=len(rows),valid_frames=len(valid),failed_frames=len(rows)-len(valid))
    if data['dataset']=='PhysTwin':
        manual=data['evaluation']['manual'];v0=np.isfinite(manual[0]).all(1);_,idx=cKDTree(pred[0]).query(manual[0,v0]);errors=[]
        for t in range(start,min(end,len(pred))):
            truth=manual[t,v0];v=np.isfinite(truth).all(1)&np.isfinite(pred[t,idx]).all(1)
            if v.any():errors.append(float(np.linalg.norm(pred[t,idx[v]]-truth[v],axis=1).mean()*1000))
        summary['manual_track_mean_mm']=float(np.mean(errors)) if errors else None
    return dict(summary=summary,frames=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['phystwin','softvt'],required=True);p.add_argument('--case',default='single_lift_sloth')
    p.add_argument('--nodes',type=int,default=100);p.add_argument('--subset',default='object-soft');p.add_argument('--task',type=int,default=0);p.add_argument('--episode',type=int,default=0)
    p.add_argument('--native-tag');p.add_argument('--tag',default='pilot');p.add_argument('--development',action='store_true');a=p.parse_args()
    data=phystwin(a.case,a.nodes) if a.dataset=='phystwin' else softvt(a.subset,a.task,a.episode)
    out=BASE/'fits'/a.tag/data['name'];fit=json.loads((out/'fit.json').read_text());raw=np.load(out/'prediction.npz')['predicted']
    predictions=baselines(data,fit['fit_end']);predictions['material_world_fem']=expand_prediction(data,raw)
    if data['dataset']=='PhysTwin':
        path=ROOT/'runs/third_party/PhysTwin/reference/experiments'/a.case/'inference.pkl'
        saved=pickle.load(open(path,'rb'));predictions['official_phystwin']=np.asarray(saved)@data['evaluation']['transform']
    if a.native_tag:
        native=BASE/'native'/a.native_tag/data['name']/'prediction.npz'
        predictions['adapted_phystwin']=np.load(native)['predicted']
    start=fit['fit_end'] if a.development else data['train_end'];end=data['train_end'] if a.development else len(data['times'])
    results={method:score(data,pred,start,end) for method,pred in predictions.items()}
    segments={}
    if data['dataset']=='SoftVTBench':
        for label,selected in [('held_grasp',data['attachment_active'].sum(1)>0),('released',(np.arange(len(data['times']))>data['grasp'])&(data['gripper_command']>0))]:
            segments[label]={method:score(data,pred,start,end,selected) for method,pred in predictions.items()}
    write_json(out/('development-metrics.json' if a.development else 'test-metrics.json'),dict(name=data['name'],interval=[start,end],results=results,segments=segments,
        note='Official PhysTwin was trained on its full official prefix; development-subprefix comparison is not observation-budget matched.' if a.development and data['dataset']=='PhysTwin' else 'Matched prefix for fitted methods; official released PhysTwin checkpoint uses author preprocessing.'))
    print(json.dumps({k:v['summary'] for k,v in results.items()},indent=2))
