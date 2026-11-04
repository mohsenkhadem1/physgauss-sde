"""Frozen-checkpoint evaluation; no optimizer or test-time fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np,torch
from .protocol import ROOT,BASE,REPORT,sha,write,partition
from .data import Sequence,batch
from .model import SceneQueryModel

def main():
    torch.set_num_threads(2);plan=json.load(open(REPORT/'protocol.json'));models={};digests={}
    for method in plan['methods']:
        folder=BASE/method;complete=json.load(open(folder/'complete.json'));assert complete['completed']
        assert sha(folder/'model.pt')==complete['checkpoint_sha256']
        m=SceneQueryModel(method=='image_blind').cuda().eval();m.load_state_dict(torch.load(folder/'model.pt',weights_only=False)['state_dict']);models[method]=m;digests[method]=complete['checkpoint_sha256']
    rows=[]
    for group,cases in plan['cases'].items():
        for case in cases:
            s=Sequence(case);out=BASE/'predictions'/case;out.mkdir(parents=True,exist_ok=True);Q=len(s.labels['query_uv']);predictions={m:[]for m in models};sourcepred={m:[]for m in models}
            for t in range(s.meta['frames']):
                x,_=batch([s.sample(t,np.arange(Q))],'cuda')
                for method,model in models.items():
                    with torch.no_grad():pred,source=model(**x)
                    predictions[method].append(pred[0].cpu().numpy());sourcepred[method].append(source[0].cpu().numpy())
            predictions={m:np.asarray(p)for m,p in predictions.items()};sourcepred={m:np.asarray(p)for m,p in sourcepred.items()}
            np.savez_compressed(out/'predictions.npz',**predictions,**{m+'_source':p for m,p in sourcepred.items()})
            predictions['privileged_initial_persistence']=np.broadcast_to(s.labels['tracks'][0],s.labels['tracks'].shape)
            result={};start=s.meta['train_end'] if group=='train' else 1
            for method,pred in predictions.items():
                assert np.isfinite(pred).all();perframe=[]
                for t in range(start,s.meta['frames']):
                    valid=s.labels['valid'][t]
                    if not valid.any():continue
                    y=s.labels['tracks'][t,valid];p=pred[t,valid]
                    source=sourcepred[method][t,valid] if method in sourcepred else s.labels['tracks'][0,valid]
                    yc=(y-s.C[2,:3,3])@s.C[2,:3,:3];pc=(p-s.C[2,:3,3])@s.C[2,:3,:3]
                    assert (yc[:,2]>0).all() and (pc[:,2]>0).all(),'Behind-camera prediction'
                    yp=yc@s.K[2].T;pp=pc@s.K[2].T
                    perframe.append(dict(frame=t,points=int(valid.sum()),tracking_mm=float(np.linalg.norm(p-y,axis=-1).mean()*1000),
                        displacement_mm=float(np.linalg.norm((p-source)-(y-s.labels['tracks'][0,valid]),axis=-1).mean()*1000),
                        held_camera_reprojection_px=float(np.linalg.norm(pp[:,:2]/pp[:,2,None]-yp[:,:2]/yp[:,2,None],axis=-1).mean())))
                result[method]=dict(frames=len(perframe),**{k:float(np.mean([r[k]for r in perframe])) for k in ('tracking_mm','displacement_mm','held_camera_reprojection_px')})
            rows.append(dict(case=case,partition=group,evaluation_start=start,results=result,prediction_sha256=sha(out/'predictions.npz')));print(case,result,flush=True)
    summary={g:{m:{k:float(np.mean([r['results'][m][k]for r in rows if r['partition']==g]))for k in ('tracking_mm','displacement_mm','held_camera_reprojection_px')}for m in list(models)+['privileged_initial_persistence']}for g in plan['cases']}
    write(REPORT/'results.json',dict(complete=True,rows=rows,summary=summary,checkpoint_sha256=digests,
        scope='Observed-video reconstruction with known cameras and annotation-selected source queries. Equal-frame then equal-case means. Track labels are reconstructed, not physical metrology. Held camera RGB was not input, but track pseudo labels may use it. Not forecasting, learned mechanics, automatic segmentation, or RGB novel-view quality.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
