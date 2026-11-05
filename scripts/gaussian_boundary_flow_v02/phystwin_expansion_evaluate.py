"""Separate post-cohort evaluator and failure-aware retrospective summary."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.phystwin_expansion import check_plan,terminal_gate,CASES
from scripts.gaussian_boundary_flow_v02.validation_cohort import cohort_lock,require_inactive,process_identity,launch,TERMINAL,write
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha

STAGES=('baseline','physics','learned_contact','retrieval','retrieval_contact')
METHODS={'persistence':('baseline','prefix_persistence'),'velocity':('baseline','prefix_velocity'),'rigid_contact':('baseline','rigid_contact'),
         'flow_raw':('physics','flow_raw'),'mixture_raw':('physics','mixture_raw'),'flow_contact':('learned_contact','flow_pose'),
         'mixture_contact':('learned_contact','mixture_pose'),'nearest_raw':('retrieval','nearest_raw'),'nearest_contact':('retrieval_contact','nearest_contact')}


def verify_score(folder,stage):
    plan,seal,complete,_=load_sealed(folder/stage)
    score=json.loads((folder/(stage+'-scores.json')).read_text())
    if score.get('case')!=plan['case']or score.get('interval')!=[seal['train_end'],complete['frames']]:raise ValueError('Score case or interval changed')
    if score.get('nodes')!=plan['args']['nodes']or score.get('baseline_budget',{}).get('frames')!=plan['prefix_frames']or score.get('baseline_budget',{}).get('point_ids')!=plan['point_ids']:raise ValueError('Score input budget changed')
    matches=[v for k,v in score.get('source_sha256',{}).items()if k.endswith('/prediction.npz')]
    if matches!=[sha(folder/stage/'prediction.npz')]:raise ValueError('Score does not bind to sealed prediction')
    return score


def evaluate(root,plan_path,plan_digest):
    root=Path(root);plan=check_plan(plan_path,plan_digest)
    with cohort_lock(root):
        rows=terminal_gate(root,plan_digest)
        for path in root.glob('*/*-evaluation-status.json'):
            old=json.loads(path.read_text())
            if old.get('state')not in TERMINAL:require_inactive(old)
        gate={'plan_sha256':plan_digest,'all_four_terminal':True,'cases':rows};gatepath=root/'evaluation-gate.json'
        if gatepath.exists()and json.loads(gatepath.read_text())!=gate:raise ValueError('Changed cohort after evaluation admission')
        # Verify every available prediction and cross-channel identity before opening any target.
        for job,row in zip(plan['cases'],rows):
            folder=root/job['case'];reference=None
            for stage in STAGES:
                if row['stages'][stage]['state']!='complete':continue
                p,s,c,a=load_sealed(folder/stage)
                identity=(p['case'],p['input_digest'],p['prefix_frames'],p['point_ids'],s['train_end'],c['frames'],p['args']['nodes'])
                if p['case']!=job['case']or p['input_digest']!=job['input_digest']or p['input_package_manifest_sha256']!=job['manifest_sha256']:raise ValueError('Prediction differs from frozen case inputs')
                if reference is not None:
                    if identity!=reference[0]or any(not np.array_equal(a[k],reference[1][k])for k in ('nodes','observed_initial','observation_matrix')):raise ValueError('Unequal prediction channels')
                else:reference=(identity,a)
                status=folder/(stage+'-evaluation-status.json')
                if status.exists():
                    old=json.loads(status.read_text())
                    if old.get('plan_sha256')!=plan_digest:raise ValueError('Changed evaluation plan')
                    if old.get('state')=='complete':verify_score(folder,stage)
        write(gatepath,gate)
        for row in rows:
            folder=root/row['case']
            for stage in STAGES:
                path=folder/(stage+'-evaluation-status.json')
                if path.exists():
                    old=json.loads(path.read_text())
                    if old.get('state')in TERMINAL:continue
                    require_inactive(old)
                    try:verify_score(folder,stage);recovered=True
                    except (OSError,ValueError,KeyError):recovered=False
                    write(path,{**old,'state':'complete'if recovered else'interrupted','recovered_verified_score':recovered});continue
                if row['stages'][stage]['state']!='complete':write(path,{'state':'blocked','plan_sha256':plan_digest,'prediction_stage':row['stages'][stage]});continue
                command=['runs/phystwin-env/bin/python','-m','scripts.gaussian_boundary_flow_v02.object_score','--run',str(folder/stage),'--output',str(folder/(stage+'-scores.json'))]
                state={'state':'running','coordinator':process_identity(),'command':command,'plan_sha256':plan_digest,'cohort_gate_sha256':sha(gatepath)};write(path,state)
                def started(identity):state['child']=identity;write(path,state)
                try:
                    code=launch(command,folder/(stage+'-evaluation.log'),started,900)
                    if code:raise RuntimeError('Evaluator exit '+str(code))
                    verify_score(folder,stage);state['state']='complete'
                except Exception as error:state.update(state='failed',error=str(error))
                write(path,state)


def qualified(summary):
    return (summary.get('total_frames',0)>0 and summary.get('valid_frames')==summary['total_frames']and summary.get('failed_frames')==0 and summary.get('no_reference_frames')==0
            and all(isinstance(summary.get(k),(int,float))and np.isfinite(summary[k])for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm')))


def summarize(root,plan_path,plan_digest,output):
    root=Path(root);check_plan(plan_path,plan_digest);rows=terminal_gate(root,plan_digest);cases=[];source_scores={}
    gate=root/'evaluation-gate.json'
    if not gate.exists()or json.loads(gate.read_text())!={'plan_sha256':plan_digest,'all_four_terminal':True,'cases':rows}:raise PermissionError('Summary requires the admitted cohort evaluation barrier')
    for case in CASES:
        folder=root/case;documents={};stages={}
        for stage in STAGES:
            p=folder/(stage+'-evaluation-status.json');status=json.loads(p.read_text())if p.exists()else{'state':'missing'};stages[stage]=status['state']
            if p.exists()and status.get('plan_sha256')!=plan_digest:raise ValueError('Evaluation status belongs to another plan')
            if status['state']=='complete':
                documents[stage]=verify_score(folder,stage);source_scores[str(folder/(stage+'-scores.json'))]=sha(folder/(stage+'-scores.json'))
        methods={}
        for name,(stage,channel)in METHODS.items():
            value=documents.get(stage,{}).get('methods',{}).get(channel,{}).get('incoming_v22',{}).get('summary')
            methods[name]={'complete':qualified(value),'metrics':value}if value else{'complete':False,'metrics':None,'status':stages[stage]}
        cases.append({'case':case,'stages':stages,'methods':methods})
    aggregates={}
    for name in METHODS:
        completed=[r['methods'][name]['metrics']for r in cases if r['methods'][name]['complete']]
        aggregates[name]={'complete_cases':len(completed),'expected_cases':4,'failed_or_missing_cases':4-len(completed),
                          **{'conditional_'+metric:float(np.mean([s[metric]for s in completed]))if completed else None for metric in ('coordinate_rmse_mm','aligned_shape_rmse_mm')}}
    pairs={}
    for candidate in ('flow_contact','mixture_contact'):
        pairs[candidate]={}
        for reference in ('nearest_contact','rigid_contact','persistence','velocity'):
            rows=[]
            for row in cases:
                a,b=row['methods'][candidate],row['methods'][reference]
                rows.append({'case':row['case'],'complete':a['complete']and b['complete'],**{metric:a['metrics'][metric]-b['metrics'][metric]if a['complete']and b['complete']else None for metric in ('coordinate_rmse_mm','aligned_shape_rmse_mm')}})
            valid=[r for r in rows if r['complete']]
            pairs[candidate][reference]={'expected_pairs':4,'complete_pairs':len(valid),'differences':rows,**{'conditional_mean_'+m:float(np.mean([r[m]for r in valid]))if valid else None for m in ('coordinate_rmse_mm','aligned_shape_rmse_mm')},'interpretation':'Positive differences mean the learned method is worse; no failed case is dropped from the denominator.'}
    result={'scope':'Four historically exposed PhysTwin plush cases; equal-case conditional means, no fresh-test or global superiority claim','plan_sha256':plan_digest,'source_scores_sha256':source_scores,'cases':cases,'aggregates':aggregates,'paired_differences':pairs,
            'all_evaluations_terminal':all(s in TERMINAL for r in cases for s in r['stages'].values()),'limitations':['Approximate pose/contact, not coupled physical inference','Different admitted node counts are matched within each case','Report all failure denominators and both coordinate and shape differences']}
    write(Path(output),result)
    lines=[result['scope']]
    for name,metrics in aggregates.items():lines.append(f"{name}: {metrics['complete_cases']}/4 complete; coordinate {metrics['conditional_coordinate_rmse_mm']} mm; shape {metrics['conditional_aligned_shape_rmse_mm']} mm")
    Path(output).with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['evaluate','summarize'],required=True)
    for name in ('root','plan','plan-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--output');a=p.parse_args()
    if a.phase=='evaluate':evaluate(a.root,a.plan,a.plan_sha256)
    else:
        if not a.output:p.error('--output is required for summarize')
        summarize(a.root,a.plan,a.plan_sha256,a.output)
