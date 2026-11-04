"""Separate evaluator for frozen source twins on recorded new interactions."""
import argparse,json,pickle,numpy as np
from scipy.spatial import cKDTree
from .registration import PAIRS,initial_bundle
from .metrics import manual_response_metrics
from .development_run import ROOT,BASE,write,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='original-frozen');args=parser.parse_args()
    registration=json.loads((ROOT/'reports/real-transfer-v20/initial-registration.json').read_text())['pairs']
    rows=[]
    for source,target in PAIRS:
        directory=ROOT/'runs/third_party/PhysTwin/data/different_types'/target
        with (directory/'gt_track_3d.pkl').open('rb') as f:manual=np.asarray(pickle.load(f))@np.diag([1.,-1.,-1.])
        manual=manual[:,np.isfinite(manual[0]).all(1)]
        split=json.loads((directory/'split.json').read_text());predictions={};provenance={};missing=[]
        for method in ('neural','analytic','native_frozen'):
            folder=(BASE/'native-transfer'/f'{source}--{target}') if method=='native_frozen' else (BASE/'frozen-transfer'/args.label/f'{source}--{target}'/method)
            if not (folder/'complete.json').exists():missing.append(method);continue
            meta=json.loads((folder/'complete.json').read_text());path=folder/'prediction.npz'
            if sha(path)!=meta['prediction_sha256']:raise ValueError('Prediction changed')
            with np.load(path) as f:predictions[method]=f['predicted']
            provenance[method]=meta
        if 'native_frozen' in predictions:
            predictions['source_shape_persistence']=np.broadcast_to(predictions['native_frozen'][0],predictions['native_frozen'].shape)
        row=next(r for r in registration if r['source']==source and r['target']==target)
        canonical,_,_=initial_bundle(source)
        canonical=canonical@np.array(row['rotation'])+np.array(row['translation_m'])
        common_ids=cKDTree(canonical).query(manual[0])[1]
        results={}
        for method,pred in predictions.items():
            if pred.shape[1]!=len(canonical):raise ValueError('Source point ordering/coverage mismatch')
            ids=common_ids
            results[method]=dict(initial_matching_mm=float(np.linalg.norm(pred[0,ids]-manual[0],axis=1).mean()*1000),
                whole_interaction=manual_response_metrics(pred[:,ids],manual,1,len(manual))['summary'],
                official_suffix=manual_response_metrics(pred[:,ids],manual,split['train'][1],split['test'][1])['summary'])
        rows.append(dict(source=source,target=target,registration=row,results=results,missing=missing,provenance=provenance))
    summaries={}
    for group,selected in [('all_five_diagnostic',rows),('initial_geometry_gate_passed',[r for r in rows if r['registration']['initial_geometry_gate']])]:
        summaries[group]={}
        for method in ('neural','analytic','native_frozen','source_shape_persistence'):
            subset=[r['results'][method]['whole_interaction'] for r in selected if method in r['results']]
            summaries[group][method]=dict(cases=len(subset),**{k:float(np.mean([s[k] for s in subset])) if subset else None for k in ('tracking_mm','displacement_mm','pair_distance_change_mm')},failed_frames=sum(s['failed_frames'] for s in subset))
    result=dict(label=args.label,rows=rows,summary=summaries,complete=all(not r['missing'] for r in rows),
        scope='Recorded cross-interaction transfer with frozen source material; only target initial geometry and known tool trajectories. Source fits have unequal author/adapter training budgets. All sequences historically exposed. Three pairs fail initial geometry gate; do not pool them into a clean OOD claim. Two passing pairs are only a small pilot.',
        interpretation='All methods use the same canonical-source frame-zero manual-point matching. Manual pair-distance change removes rigid motion but is a standard geometry diagnostic, not force ground truth. Frozen native controller springs use median source strength; this is an adapted baseline.')
    write(ROOT/'reports/real-transfer-v20'/f'{args.label}-transfer-scores.json',result);print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
