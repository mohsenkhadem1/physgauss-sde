"""Common held-suffix metrics for the fixed 240-node rebenchmark."""
import json,pickle,numpy as np
from scipy.spatial import cKDTree
from .development_run import ROOT,write,sha
from .refined_campaign import LABEL
from .metrics import manual_response_metrics
from scripts.real_transfer_v19.phystwin_campaign import CASES
from scripts.external_benchmark.data import phystwin
from scripts.external_benchmark.evaluate import expand_prediction,score,baselines

def main():
    rows=[]
    for case in CASES:
        data=phystwin(case,240);predictions=baselines(data,data['train_end']);missing=[];hashes={}
        for method in ('neural','analytic'):
            folder=ROOT/'runs/real-transfer-v19/fits'/LABEL/case/method;path=folder/'prediction.npz'
            if not (folder/'complete.json').exists():missing.append(method);continue
            meta=json.loads((folder/'complete.json').read_text())
            if sha(path)!=meta['prediction_sha256']:raise ValueError('Changed prediction')
            with np.load(path) as f:predictions[method]=expand_prediction(data,f['predicted'])
            hashes[method]=meta['prediction_sha256']
        native=ROOT/'runs/third_party/PhysTwin/reference/experiments'/case/'inference.pkl'
        predictions['official_phystwin']=np.asarray(pickle.load(native.open('rb')))@data['evaluation']['transform']
        manual=data['evaluation']['manual'];manual=manual[:,np.isfinite(manual[0]).all(1)]
        results={}
        for method,pred in predictions.items():
            ids=cKDTree(pred[0]).query(manual[0])[1]
            results[method]=dict(geometry=score(data,pred,data['train_end'],len(data['times']))['summary'],
                manual=manual_response_metrics(pred[:,ids],manual,data['train_end'],len(manual))['summary'])
        rows.append(dict(case=case,results=results,missing=missing,prediction_sha256=hashes))
    complete=all(not r['missing'] for r in rows);summary={}
    for method in ('neural','analytic','official_phystwin','position_ridge'):
        subset=[r['results'][method]['manual'] for r in rows if method in r['results']]
        summary[method]=dict(cases=len(subset),**{k:float(np.mean([s[k] for s in subset])) if subset else None for k in ('tracking_mm','displacement_mm','pair_distance_change_mm')})
    result=dict(rows=rows,summary=summary,complete=complete,label=LABEL,
        scope='Existing eight-sequence rebenchmark, 240 nodes selected using only development-prefix validation. Partial summaries must not be compared across unequal case sets.')
    write(ROOT/'reports/real-transfer-v20/refined-scores.json',result);print(json.dumps(dict(complete=complete,summary=summary),indent=2))

if __name__=='__main__':main()
