"""Evaluator-only scorer of sealed exposed SoftVT stable-package pilots."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from scipy.spatial import cKDTree
from scripts.gaussian_boundary_flow_v02 import object_score as shared
from scripts.gaussian_boundary_flow_v02.export_inputs import load_package
from scripts.gaussian_boundary_flow_v02.benchmark_inputs import (
    SOFTVT,predictor_digest,load_softvt_evaluation,_require_exposed,_softvt_path)


def verified_inputs(plan,seal,complete,pred,root=SOFTVT):
    match=re.fullmatch(r'(object-soft|spatial-soft)-task([012])-demo([0-9]+)',plan['case'])
    if not match:raise ValueError('Unsupported SoftVT case')
    subset,task,episode=match[1],int(match[2]),int(match[3]);_require_exposed(task,episode)
    path=Path(plan['input_package'])
    if not path.is_absolute():path=Path(__file__).resolve().parents[2]/path
    if shared.sha(path/'manifest.json')!=plan.get('input_package_manifest_sha256'):
        raise ValueError('Changed or missing sealed input package manifest digest')
    data=load_package(path)
    if data['dataset']!='SoftVTBench' or data['name']!=plan['case'] or predictor_digest(data)!=plan['input_digest']:
        raise ValueError('Changed predictor identity')
    if Path(data['source']).resolve()!=_softvt_path(subset,task,root).resolve():raise ValueError('Evaluation source differs from package source')
    if int(data['train_end'])!=int(seal['train_end']) or int(complete['fit_end'])!=int(data['train_end']):raise ValueError('Changed prefix cutoff')
    if int(data['train_end'])!=int(.6*len(data['times'])):raise ValueError('This scorer requires official60percent prefix')
    if complete['frames']!=len(data['times']) or len(data['nodes'])!=plan['args']['nodes']:raise ValueError('Changed forecast geometry/clock')
    for key in ['nodes','observation_matrix','observed_initial']:
        if not np.array_equal(pred[key],data[key]):raise ValueError('Changed predictor geometry: '+key)
    names=shared.prediction_methods(plan,pred,len(data['times']))
    # Validate baseline observation budget before evaluator access too.
    baselines=shared.prefix_baselines(data,plan['prefix_frames'],plan['point_ids'])
    return data,(subset,task,episode),names,baselines


def target_observation_ids(data,evaluation):
    """Fix node identities from initial geometry only, never future matching."""
    initial=np.asarray(evaluation['points'][0]);support=np.asarray(data.get('support_geometry',data['nodes']))
    if not np.array_equal(initial,support):raise ValueError('Evaluation initial node order differs from exported full geometry')
    distance,ids=cKDTree(initial).query(data['observed_initial'])
    if np.any(distance>1e-10) or len(np.unique(ids))!=len(ids):raise ValueError('Ambiguous or mismatched observed node identities')
    if not np.array_equal(initial[ids],data['observed_initial']):raise ValueError('Observation initial geometry mismatch')
    if not np.array_equal(evaluation['points'][:data['train_end'],ids],data['observed']):raise ValueError('Evaluation prefix differs from sealed observations')
    return ids


def score_case(run,root=SOFTVT):
    plan,seal,complete,pred=shared.load_sealed(run)
    data,case,names,baselines=verified_inputs(plan,seal,complete,pred,root)
    evaluation=load_softvt_evaluation(*case,root=root)
    ids=target_observation_ids(data,evaluation);target=evaluation['points'][:,ids];mask=evaluation['mask'][:,ids]
    start,end=data['train_end'],len(data['times'])
    if len(target)!=end:raise ValueError('Evaluation forecast clock mismatch')
    methods={name:shared.expand_points(pred[name],pred['nodes'],pred['observation_matrix'],pred['observed_initial'])for name in names}
    methods.update(baselines)
    scores={name:shared.geometry_score(values,target,mask,start,end)for name,values in methods.items()}
    intervals={name:shared.interval_score(pred.get(name+'_samples'),pred['nodes'],pred['observation_matrix'],pred['observed_initial'],target,mask,start,end)for name in names}
    for name in intervals:
        family=name.split('_')[0];diag=complete.get('forecast_samples',{}).get(family)
        if diag:
            intervals[name]['forecast_selection']=dict(requested=diag['requested'],valid=diag['valid'],failed=len(diag['failures']))
            intervals[name]['interpretation']='Coverage of solver-successful stored trajectories only; not calibrated unconditional posterior coverage. Failed trajectories remain reported.'
    diagnostic_path=Path(run)/'context-diagnostic.json'
    diagnostic=json.loads(diagnostic_path.read_text())if diagnostic_path.exists()else None
    return dict(case=data['name'],scope='Historically exposed SoftVT development, sparse observed-node suffix scoring; no rendering or population superiority claim',
                interval=[start,end],observed_node_ids=ids.tolist(),nodes=len(data['nodes']),observation_count=len(ids),
                methods=scores,posterior_intervals=intervals,forecast_samples=complete.get('forecast_samples'),
                baseline_budget={'frames':plan['prefix_frames'],'point_ids':plan['point_ids']},
                prefix_support_diagnostic=diagnostic,**shared.support_screen(diagnostic),
                source_sha256={'prediction':shared.sha(Path(run)/'prediction.npz'),'plan':shared.sha(Path(run)/'plan.json')},
                evaluator_only=True,full_mesh_metric=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output');args=parser.parse_args()
    result=score_case(args.run);out=Path(args.output)if args.output else Path(args.run)/'scores.json'
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    print(json.dumps({name:row['summary']for name,row in result['methods'].items()},indent=2))

if __name__=='__main__':main()
