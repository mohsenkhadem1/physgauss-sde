"""Verified retrospective eight-case comparison; no evaluator or model execution."""
import json
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.expansion_summary import source_paths,run_paths,validate_channel_identity,complete_metric
from scripts.gaussian_boundary_flow_v02.object_score import sha

REPORT=Path('reports/gaussian-boundary-flow-v02')
ORIGINAL=('single_lift_sloth','single_push_sloth','single_lift_zebra','double_lift_zebra')
NEW=('single_lift_dinosor','double_lift_sloth','double_stretch_sloth','double_stretch_zebra')
METHODS={'flow_raw':('learned-raw','flow_raw'),'mixture_raw':('learned-raw','mixture_raw'),
         'flow_contact':('learned-contact','flow_pose'),'mixture_contact':('learned-contact','mixture_pose'),
         'nearest_raw':('retrieval-raw','nearest_raw'),'nearest_contact':('retrieval-contact','nearest_contact'),
         'rigid_contact':('baseline','rigid_contact'),'persistence':('baseline','prefix_persistence'),'velocity':('baseline','prefix_velocity')}


def paths(case):
    if case in ORIGINAL:
        scores=source_paths(case);runs=run_paths(case)
        scores['baseline']=REPORT/'original-phys-controls-scores'/(case+'.json')
        runs['baseline']=Path('runs/gaussian-boundary-flow-v05-original-phys-controls')/case
        statuses={'baseline':REPORT/'original-phys-controls-scores'/(case+'-status.json')}
    else:
        root=Path('runs/gaussian-boundary-flow-v05-phystwin')/case
        labels={'learned-raw':'physics','learned-contact':'learned_contact','retrieval-raw':'retrieval','retrieval-contact':'retrieval_contact','baseline':'baseline'}
        scores={name:root/(stage+'-scores.json')for name,stage in labels.items()};runs={name:root/stage for name,stage in labels.items()}
        statuses={name:root/(stage+'-evaluation-status.json')for name,stage in labels.items()}
    return scores,runs,statuses


def summarize():
    cases=[];hashes={}
    for case in ORIGINAL+NEW:
        scorepaths,runs,statuses=paths(case);documents={};states={}
        for channel,path in scorepaths.items():
            status=json.loads(statuses[channel].read_text())if channel in statuses and statuses[channel].exists()else None
            states[channel]=status.get('state')if status else ('archived_score'if path.exists()else'missing')
            if status is not None and status['state']!='complete':continue
            if not path.exists():continue
            if status and status.get('score_sha256')and sha(path)!=status['score_sha256']:raise ValueError('Changed audited original-control score')
            documents[channel]=json.loads(path.read_text());hashes[str(path)]=sha(path)
        identities=validate_channel_identity(case,documents,runs)
        methods={}
        for name,(channel,key)in METHODS.items():
            value=documents.get(channel,{}).get('methods',{}).get(key,{}).get('incoming_v22',{}).get('summary')
            methods[name]={'complete':complete_metric(value),'metrics':{k:value.get(k)for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm','total_frames','valid_frames','failed_frames','no_reference_frames')}}if value else {'complete':False,'metrics':None,'status':states[channel]}
        first=next(iter(documents.values()),{});raw=documents.get('learned-raw',{})
        cases.append({'case':case,'group':'original four; rigid controls added retrospectively'if case in ORIGINAL else'four-case retrospective expansion',
                      'nodes':first.get('nodes'),'interval':first.get('interval'),'baseline_budget':first.get('baseline_budget'),
                      'verified_source_channels':identities,'channel_states':states,'methods':methods,
                      'forecast_samples':{name:{'requested':x.get('requested'),'valid':x.get('valid'),'failed':len(x.get('failures',[]))}for name,x in (raw.get('forecast_samples')or{}).items()}})
    aggregates={}
    for name in METHODS:
        valid=[r['methods'][name]['metrics']for r in cases if r['methods'][name]['complete']]
        aggregates[name]={'complete_cases':len(valid),'expected_cases':8,'failed_or_missing_cases':8-len(valid),
                          **{'conditional_mean_'+k:float(np.mean([r[k]for r in valid]))if valid else None for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm')}}
    paired={}
    for candidate in ('flow_contact','mixture_contact'):
        paired[candidate]={}
        for reference in ('rigid_contact','nearest_contact','persistence','velocity'):
            diffs=[]
            for case in cases:
                a,b=case['methods'][candidate],case['methods'][reference];valid=a['complete']and b['complete']
                diffs.append({'case':case['case'],'complete':valid,**{k:a['metrics'][k]-b['metrics'][k]if valid else None for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm')}})
            valid=[d for d in diffs if d['complete']]
            paired[candidate][reference]={'expected_pairs':8,'complete_pairs':len(valid),'per_case_differences':diffs,
                **{'conditional_mean_difference_'+k:float(np.mean([d[k]for d in valid]))if valid else None for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm')},
                'coordinate_better_cases':sum(d['coordinate_rmse_mm']<0 for d in valid),'coordinate_worse_cases':sum(d['coordinate_rmse_mm']>0 for d in valid),'interpretation':'Positive differences mean the learned method is worse; missing and failed pairs remain in the eight-case denominator.'}
    result={'scope':'Eight historically exposed PhysTwin plush cases; original-four rigid controls are retrospective additions, not a new untouched test.',
            'aggregation':'Equal-case means, complete-case conditional when necessary; incoming-v22 mask only. Geometry and observation budgets verified within each case.',
            'all_required_methods_complete':all(v['complete_cases']==8 for v in aggregates.values()),'cases':cases,'aggregates':aggregates,'paired_differences':paired,'score_sources_sha256':hashes,
            'control_and_expansion_plan_sha256':{str(p):sha(p) for p in (REPORT/'original-phys-controls-plan.json', REPORT/'phystwin-v05-expansion-plan.json') if p.exists()},
            'limitations':['No outcome-driven model selection or per-case winner mixture','64/119/120 node discretizations differ across cases but match all comparators within each case','Pose/contact is an approximate rigid correction, not coupled or calibrated physical inference','Failed posterior draws remain reported; a valid mean trajectory does not establish unconditional uncertainty calibration']}
    output=REPORT/'phystwin-all-eight-comparison.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=[result['scope'],result['aggregation'],'']
    for name,row in aggregates.items():lines.append(f"{name}: {row['complete_cases']}/8 complete; coordinate {row['conditional_mean_coordinate_rmse_mm']} mm; shape {row['conditional_mean_aligned_shape_rmse_mm']} mm")
    lines+=['','Paired learned-minus-reference changes; positive means worse.']
    for candidate,comparisons in paired.items():
        for reference,row in comparisons.items():lines.append(f"{candidate} vs {reference}: {row['complete_pairs']}/8 pairs; coordinate {row['conditional_mean_difference_coordinate_rmse_mm']} mm; shape {row['conditional_mean_difference_aligned_shape_rmse_mm']} mm")
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':summarize()
