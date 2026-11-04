"""Separate evaluator; never imported by the fitting worker."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import baselines,expand_prediction,score
from scripts.real_transfer_v19.phystwin_run import ROOT,BASE,sha,write
from scripts.real_transfer_v19.phystwin_campaign import CASES,LABEL

def extrapolation(data):
    # Known commands only. This is input-range extrapolation, not unseen-object OOD.
    controls=data['controller_targets'].mean(2).reshape(len(data['times']),-1)
    end=data['train_end'];margin=.01*np.ptp(data['nodes'],axis=0).max()
    return ((controls<controls[:end].min(0)-margin)|(controls>controls[:end].max(0)+margin)).any(1)

def manual_frame_errors(data,pred,start,selected=None):
    from scipy.spatial import cKDTree
    manual=data['evaluation']['manual'];v0=np.isfinite(manual[0]).all(1)
    if not np.isfinite(pred[0]).all():return []
    ids=cKDTree(pred[0]).query(manual[0,v0])[1];values=[]
    for frame in range(start,min(len(pred),len(manual))):
        if selected is not None and not selected[frame]:continue
        target=manual[frame,v0];valid=np.isfinite(target).all(1)
        if valid.any() and np.isfinite(pred[frame,ids[valid]]).all():
            values.append(float(np.linalg.norm(pred[frame,ids[valid]]-target[valid],axis=1).mean()*1000))
    return values

def evaluate():
    rows=[]
    for case in CASES:
        data=phystwin(case,120);predictions=baselines(data,data['train_end']);sources={}
        original=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
        predictions['official_phystwin']=np.asarray(pickle.load(original.open('rb')))@data['evaluation']['transform']
        sources[str(original.relative_to(ROOT))]=sha(original)
        missing=[]
        for method in ['neural','analytic']:
            folder=BASE/'fits'/LABEL/case/method
            if not (folder/'complete.json').exists():
                missing.append(method);continue
            meta=json.loads((folder/'complete.json').read_text());path=folder/'prediction.npz'
            if sha(path)!=meta['prediction_sha256']:raise ValueError('Prediction hash mismatch')
            with np.load(path) as a:predictions[method]=expand_prediction(data,a['predicted'])
            sources[str(path.relative_to(ROOT))]=sha(path)
        end=data['train_end'];ood=extrapolation(data);metrics={}
        for method,pred in predictions.items():
            full=score(data,pred,end,len(data['times']))['summary']
            selected=score(data,pred,end,len(data['times']),ood)['summary']
            # Legacy score's manual branch ignores selected; replace with explicit filtered score.
            values=manual_frame_errors(data,pred,end,ood)
            selected['manual_track_mean_mm']=float(np.mean(values)) if values else None
            selected['manual_valid_frames']=len(values)
            metrics[method]=dict(full_suffix=full,input_range_extrapolation=selected)
        rows.append(dict(case=case,train_end=end,test_frames=len(data['times'])-end,
                         extrapolation_frames=int(ood[end:].sum()),results=metrics,missing_methods=missing,source_sha256=sources))
    summaries={}
    for segment in ['full_suffix','input_range_extrapolation']:
        summaries[segment]={}
        for method in ['neural','analytic','official_phystwin','position_ridge','persistence','control_rigid']:
            available=[r['results'][method][segment] for r in rows if method in r['results']]
            summaries[segment][method]={key:float(np.mean([m[key] for m in available if m.get(key) is not None]))
                for key in ['manual_track_mean_mm','coordinate_rmse_mm','aligned_shape_rmse_mm']
                if any(m.get(key) is not None for m in available)}
            summaries[segment][method].update(cases=len(available),valid_frames=sum(m['valid_frames'] for m in available),total_frames=sum(m['total_frames'] for m in available))
    result=dict(rows=rows,summary=summaries,complete=all(not r['missing_methods'] for r in rows),
                interpretation='Exploratory historical PhysTwin rebenchmark. Equal sequence means. Input-range extrapolation is computed from known tool commands, not object outcomes; not novel-object or interaction-family OOD. Native released PhysTwin has author preprocessing/optimization; matched analytic control shares our geometry, prefix and solver budget.')
    write(ROOT/'reports/real-transfer-v19'/(LABEL+'-scores.json'),result)
    print(json.dumps(dict(complete=result['complete'],summary=summaries),indent=2))

if __name__=='__main__':evaluate()
