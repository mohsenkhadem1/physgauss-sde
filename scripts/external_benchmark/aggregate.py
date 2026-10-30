"""Score every preregistered sequence; preserve failed jobs and numerical failures."""
import sys,json,pickle,argparse,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.external_benchmark.common import *
from scripts.external_benchmark.data import phystwin,softvt
from scripts.external_benchmark.evaluate import baselines,expand_prediction,score
from scripts.external_benchmark.calibrated_baseline import calibrated_rigid

def evaluate(data):
    fit_end=data['train_end'];end=len(data['times']);predictions=baselines(data,fit_end);predictions['prefix_calibrated_rigid']=calibrated_rigid(data)
    paths={};missing=[];diagnostics={}
    variants=[('primary','material_world_fem'),('primary-no-field','no_world_potential_fem')]
    if data['dataset']=='SoftVTBench':variants += [('adaptive','adaptive_material_world_fem'),('adaptive-no-field','adaptive_no_world_potential_fem'),('physical-release','physical_release_material_world_fem'),('physical-release-no-field','physical_release_no_world_potential_fem')]
    for tag,method in variants:
        folder=BASE/'fits'/tag/data['name'];path=folder/'prediction.npz'
        if path.exists():
            predictions[method]=expand_prediction(data,np.load(path)['predicted']);paths[method]=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
            fit=json.loads((folder/'fit.json').read_text());prediction=json.loads((folder/'prediction.json').read_text())
            diagnostics[method]=dict(fit_rmse_mm=fit['best']['fit_rmse_mm'],optimizer=fit['optimizer'],elapsed_seconds=fit['elapsed_seconds'],failed_frames=prediction['failed_frames'],theta=fit['best']['theta'])
        else:missing.append(method)
    if data['dataset']=='PhysTwin':
        path=ROOT/'runs/third_party/PhysTwin/reference/experiments'/data['name']/'inference.pkl'
        predictions['official_phystwin']=np.asarray(pickle.load(open(path,'rb')))@data['evaluation']['transform']
        paths['official_phystwin']=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
    else:
        path=BASE/'native/primary'/data['name']/'prediction.npz'
        if path.exists():
            predictions['adapted_phystwin']=np.load(path)['predicted'];paths['adapted_phystwin']=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))
        else:missing.append('adapted_phystwin')
    results={method:score(data,pred,fit_end,end) for method,pred in predictions.items()}
    segments={}
    if data['dataset']=='SoftVTBench':
        for label,selected in [('held_grasp',data['attachment_active'].sum(1)>0),('released',(np.arange(end)>data['grasp'])&(data['gripper_command']>0))]:
            segments[label]={method:score(data,pred,fit_end,end,selected) for method,pred in predictions.items()}
    # Quantify actual deformation independently of prediction success.
    deformation=score(data,np.broadcast_to(data['evaluation']['points'][0],data['evaluation']['points'].shape),fit_end,end)['summary']['aligned_shape_rmse_mm']
    controls=data['controls'].reshape(end,-1);lo=controls[:fit_end].min(0);hi=controls[:fit_end].max(0)
    outside=((controls[fit_end:]<lo-.001)|(controls[fit_end:]>hi+.001)).any(1)
    return dict(control_outside_prefix_box_fraction=float(outside.mean()),name=data['name'],dataset=data['dataset'],interval=[fit_end,end],initial_nodes=len(data['nodes']),results=results,segments=segments,missing=missing,prediction_files=paths,fit_diagnostics=diagnostics,actual_nonrigid_shape_rmse_mm=deformation,nonrigid_to_total_motion_ratio=deformation/max(results['persistence']['summary']['coordinate_rmse_mm'],1e-12))

def summarize(rows,segment=None):
    methods=sorted(set(m for row in rows for m in (row['results'] if segment is None else row['segments'].get(segment,{}))))
    result={}
    for method in methods:
        summaries=[(row['results'] if segment is None else row['segments'].get(segment,{})).get(method,{}).get('summary',{}) for row in rows]
        keys=sorted(set(k for s in summaries for k in s if k.endswith('_mm')))
        values={k:[s[k] for s in summaries if s.get(k) is not None] for k in keys}
        result[method]=dict(sequences_expected=len(rows),sequences_present=sum(bool(s) for s in summaries),sequences_complete=sum(bool(s) and s.get('failed_frames',0)==0 for s in summaries),total_frames=sum(s.get('total_frames',0) for s in summaries),failed_frames=sum(s.get('failed_frames',0) for s in summaries),mean={k:float(np.mean(v)) for k,v in values.items() if v},median={k:float(np.median(v)) for k,v in values.items() if v})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    frozen=json.loads((REPORT/'frozen-protocol.json').read_text())
    for source,expected in frozen['source_sha256'].items():
        if sha256(ROOT/source)!=expected:raise RuntimeError(f'Frozen prediction/scoring source changed: {source}')
    statuses=[json.loads(p.read_text()) for p in (BASE/'primary-jobs').glob('*.json')]
    if not a.partial and (len(statuses)<len(frozen['jobs']) or any(s['state'] not in ['complete','failed'] for s in statuses)):raise RuntimeError('Primary queue is not finished')
    rows=[]
    for case in frozen['phystwin']['cases']:
        rows.append(evaluate(phystwin(case,300)));print('scored',case,flush=True)
    for subset in frozen['softvt']['subsets']:
        for task in frozen['softvt']['tasks']:
            for ep in frozen['softvt']['episodes']:
                data=softvt(subset,task,ep);rows.append(evaluate(data));print('scored',data['name'],flush=True)
    summaries={}
    for dataset in ['PhysTwin','SoftVTBench']:
        selected=[r for r in rows if r['dataset']==dataset];summaries[dataset]=dict(full=summarize(selected))
        if dataset=='SoftVTBench':
            for segment in ['held_grasp','released']:summaries[dataset][segment]=summarize(selected,segment)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),partial=a.partial,aggregation='Equal weight per sequence; frame means within each sequence. Errors conditional on numerically valid frames; failures and missing jobs retained explicitly. No confidence claims from correlated frames.',summaries=summaries,sequences=rows,job_status=statuses)
    write_json((BASE/'partial-results.json') if a.partial else (REPORT/'results.json'),result)
    if not a.partial:(REPORT/'results.json.gz').write_bytes(gzip.compress((REPORT/'results.json').read_bytes(),compresslevel=9,mtime=0))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
