"""Replay common manual-response metrics and audit existing FEM meshes/fits."""
import json,pickle,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .metrics import manual_response_metrics,tetra_sanity
from .development_run import ROOT,write
from scripts.real_transfer_v19.phystwin_campaign import CASES
from scripts.real_transfer_v19.phystwin_run import load
from scripts.real_transfer_v19.neural_fem import NeuralFEM,parameters
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction,baselines

def main():
    rows=[]
    for case in CASES:
        data=phystwin(case,120);prefix,_=load(case,120,False);model=NeuralFEM(prefix,'cpu')
        e=model.elements.numpy();a=np.repeat(e,4,axis=1).ravel();b=np.tile(e,(1,4)).ravel()
        count,labels=connected_components(coo_matrix((np.ones(len(a)),(a,b)),shape=(120,120)))
        manual=data['evaluation']['manual'];visible=np.isfinite(manual[0]).all(1);manual=manual[:,visible]
        predictions=baselines(data,data['train_end']);extras={}
        for method in ('neural','analytic'):
            folder=ROOT/'runs/real-transfer-v19/fits/exploratory-01'/case/method
            with np.load(folder/'prediction.npz') as f:raw=f['predicted']
            predictions[method]=expand_prediction(data,raw)
            fit=json.loads((folder/'fit.json').read_text());plan=json.loads((folder/'plan.json').read_text())
            at_bounds=[i for i,(x,(low,high)) in enumerate(zip(fit['best']['parameters'],plan['bounds']))
                       if high>low and min(abs(x-low),abs(x-high))<1e-4]
            extras[method]=dict(mesh=tetra_sanity(data['nodes'],raw,e),connected_components=int(count),
                component_sizes=np.bincount(labels).tolist(),optimizer=fit['optimizer'],
                selected_prefix_coordinate_rmse_mm=fit['best']['coordinate_rmse_mm'],
                parameters_at_nonfixed_bounds=at_bounds)
        native=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
        predictions['official_phystwin']=np.asarray(pickle.load(native.open('rb')))@data['evaluation']['transform']
        results={}
        for method,pred in predictions.items():
            ids=cKDTree(pred[0]).query(manual[0])[1]
            results[method]=manual_response_metrics(pred[:,ids],manual,data['train_end'],len(manual))
        rows.append(dict(case=case,metrics=results,diagnostics=extras))
    methods=list(rows[0]['metrics']);summary={}
    for method in methods:
        entries=[r['metrics'][method]['summary'] for r in rows]
        summary[method]={k:float(np.mean([r[k] for r in entries])) for k in ('tracking_mm','displacement_mm','pair_distance_change_mm')}
        summary[method]['failed_frames']=sum(r['failed_frames'] for r in entries)
    result=dict(rows=rows,summary=summary,
        scope='Post-hoc diagnosis of exposed sequences. Neither manual pair-distance change nor numerical validity is a new metric or proof of force accuracy.',
        renderer_status='Raw camera images and Gaussian appearance assets are not part of the previously downloaded evaluation subset; no rendering score fabricated.')
    write(ROOT/'reports/real-transfer-v20/existing-audit.json',result);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
