"""Failure-aware fixed-case expansion summary; no selection of per-case winners."""
import json,hashlib
from pathlib import Path
import numpy as np
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha

REPORT=Path('reports/gaussian-boundary-flow-v02')
METHODS={
 'flow_raw':('learned-raw','flow_raw'),'mixture_raw':('learned-raw','mixture_raw'),
 'flow_pose_contact':('learned-contact','flow_pose'),'mixture_pose_contact':('learned-contact','mixture_pose'),
 'nearest_bank_raw':('retrieval-raw','nearest_raw'),'nearest_bank_pose_contact':('retrieval-contact','nearest_contact'),
 'prefix_persistence':('learned-raw','prefix_persistence'),'prefix_velocity':('learned-raw','prefix_velocity')}


def complete_metric(value):
    counts=[value.get(k)for k in ('total_frames','valid_frames','failed_frames','no_reference_frames')]
    return (all(isinstance(v,int)and not isinstance(v,bool)and v>=0 for v in counts)
            and counts[0]>0 and counts[0]==counts[1]and counts[2]==counts[3]==0
            and all(isinstance(value.get(k),(int,float))and np.isfinite(value[k])for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm')))


def qualified(value):
    return value.get('complete',False)and all(isinstance(value.get(k),(int,float))and np.isfinite(value[k])for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm'))


def run_paths(case):
    existing=case in ('object-soft-task1-demo0','single_push_sloth')
    root='gaussian-boundary-flow-v03-'
    return {category:Path('runs')/(root+suffix)/case for category,suffix in
            {'learned-raw':'package'if existing else'expansion',
             'learned-contact':'pose-contact'if existing else'expansion-contact',
             'retrieval-raw':'retrieval'if existing else'expansion-retrieval',
             'retrieval-contact':'retrieval-contact'if existing else'expansion-retrieval-contact'}.items()}


def validate_channel_identity(case,documents,runs):
    """Check scores against their sealed trajectory and equal within-case inputs."""
    reference=None;identities={}
    for category,document in documents.items():
        directory=runs[category];plan,seal,complete,arrays=load_sealed(directory)
        if document.get('case')!=case or plan.get('case')!=case:raise ValueError('Score/source case identity mismatch')
        expected=[seal['train_end'],complete['frames']]
        if document.get('interval')!=expected or document.get('nodes')!=len(arrays['nodes']):raise ValueError('Score interval/node identity mismatch')
        budget=document.get('baseline_budget',{})
        if budget.get('frames')!=plan['prefix_frames']or budget.get('point_ids')!=plan['point_ids']:raise ValueError('Score observation budget mismatch')
        digests=document.get('source_sha256',{})
        prediction_hashes=[value for key,value in digests.items()if key=='prediction'or key.endswith('/prediction.npz')]
        if prediction_hashes!=[sha(directory/'prediction.npz')]:raise ValueError('Score is not bound to sealed prediction bytes')
        identity={'input_digest':plan['input_digest'],'prefix_frames':plan['prefix_frames'],'point_ids':plan['point_ids'],
                  'interval':expected,'nodes':len(arrays['nodes']),'scoring_mask':document.get('scoring_mask'),
                  'observed_node_ids':document.get('observed_node_ids')}
        if reference is not None:
            if identity!=reference[0]:raise ValueError('Cross-channel input/evaluation identity mismatch')
            for key in ('nodes','observation_matrix','observed_initial'):
                if not np.array_equal(arrays[key],reference[1][key]):raise ValueError('Cross-channel geometry mismatch: '+key)
        else:reference=(identity,arrays)
        identities[category]={'run':str(directory),'prediction_sha256':prediction_hashes[0],'plan_sha256':sha(directory/'plan.json'),'input_digest':plan['input_digest']}
    return identities


def aggregate(rows,expected):
    out={}
    for method in METHODS:
        valid=[r['methods'][method]for r in rows if method in r['methods'] and qualified(r['methods'][method])]
        out[method]={'completed_cases':len(valid),'expected_cases':expected,
          'conditional_coordinate_rmse_mm':float(np.mean([r['coordinate_rmse_mm']for r in valid]))if valid else None,
          'conditional_shape_rmse_mm':float(np.mean([r['aligned_shape_rmse_mm']for r in valid]))if valid else None,
          'scope':'Equal-case mean on complete cases; conditional if any expected case missing or failed'}
    decisions={}
    base='nearest_bank_pose_contact'
    for method in ('flow_pose_contact','mixture_pose_contact'):
        pairs=[(r['methods'][method],r['methods'][base])for r in rows if all(k in r['methods'] and qualified(r['methods'][k])for k in (method,base))]
        diffs=[a['coordinate_rmse_mm']-b['coordinate_rmse_mm']for a,b in pairs]
        evaluable=len(rows)==expected and len(pairs)==expected
        decisions[method]={'paired_complete_cases':len(pairs),'wins':sum(x<0 for x in diffs),
            'paired_differences_mm':diffs,'gate_evaluable':evaluable,
            'dataset_gate_pass':evaluable and float(np.mean(diffs))<0 and sum(x<0 for x in diffs)>=2,
            'note':'Narrow frozen nearest-bank-contact criterion only; not a globally best predictor claim. Missing/failed required comparisons block promotion.'}
    strong={}
    for method in ('flow_pose_contact','mixture_pose_contact'):
        strong[method]={}
        for baseline in ('prefix_persistence','prefix_velocity'):
            pairs=[(r['methods'][method],r['methods'][baseline])for r in rows if all(k in r['methods']and qualified(r['methods'][k])for k in (method,baseline))]
            diffs=[a['coordinate_rmse_mm']-b['coordinate_rmse_mm']for a,b in pairs]
            shape=[a['aligned_shape_rmse_mm']-b['aligned_shape_rmse_mm']for a,b in pairs]
            strong[method][baseline]={'paired_complete_cases':len(pairs),'expected_cases':expected,
                'paired_coordinate_deficits_mm':diffs,'mean_coordinate_deficit_mm':float(np.mean(diffs))if diffs else None,
                'paired_shape_deficits_mm':shape,'mean_shape_deficit_mm':float(np.mean(shape))if shape else None,
                'coordinate_worse_cases':sum(d>0 for d in diffs),'coordinate_better_cases':sum(d<0 for d in diffs),
                'all_required_pairs_complete':len(pairs)==expected,'interpretation':'Positive deficit means candidate is worse; separate diagnostic, does not alter frozen nearest-bank criterion'}
    return {'methods':out,'candidate_gates':decisions,'strong_baseline_comparisons':strong}


def source_paths(case):
    if case in ('object-soft-task1-demo0','single_push_sloth'):
        raw=REPORT/('softvt-affine-pilot-scores.json'if case.startswith('object')else'push-sloth-affine-pilot-scores.json')
        return {'learned-raw':raw,'learned-contact':Path('runs/gaussian-boundary-flow-v03-pose-contact')/case/'scores.json',
          'retrieval-raw':Path('runs/gaussian-boundary-flow-v03-retrieval')/case/'scores.json',
          'retrieval-contact':Path('runs/gaussian-boundary-flow-v03-retrieval-contact')/case/'scores.json'}
    return {name:REPORT/'expansion-downstream'/case/(name+'-scores.json')for name in ('learned-raw','learned-contact','retrieval-raw','retrieval-contact')}


def summarize():
    plan=json.loads((REPORT/'compactexpansion-plan.json').read_text());cases=[];sources={}
    for case in plan['cases']:
        documents={}
        for category,path in source_paths(case).items():
            if path.exists():
                documents[category]=json.loads(path.read_text())
                if documents[category].get('case')!=case:raise ValueError('Score case identity mismatch: '+str(path))
                sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        identities=validate_channel_identity(case,documents,run_paths(case))
        methods={}
        for method,(category,name)in METHODS.items():
            if category not in documents or name not in documents[category]['methods']:continue
            value=documents[category]['methods'][name];value=value.get('incoming_v22',value)['summary']
            complete=complete_metric(value)
            methods[method]={k:value[k]for k in ('coordinate_rmse_mm','aligned_shape_rmse_mm','total_frames','valid_frames','failed_frames','no_reference_frames')}
            methods[method]['complete']=complete
        raw=documents.get('learned-raw',{})
        sample_validity={k:{'requested':v.get('requested'),'valid':v.get('valid'),'failed':len(v.get('failures',[]))}for k,v in (raw.get('forecast_samples')or{}).items()}
        cases.append({'case':case,'nodes':raw.get('nodes'),'forecast_samples':sample_validity,'out_of_bank_warning':raw.get('out_of_bank_warning'),'interval_diagnostics':raw.get('posterior_intervals'),'dataset':'SoftVTBench'if '-soft-'in case else'PhysTwin','methods':methods,
                      'available_score_channels':list(documents),'verified_channel_identity':identities})
    datasets={name:aggregate([r for r in cases if r['dataset']==name],4)for name in ('SoftVTBench','PhysTwin')}
    all_evaluable=all(v['candidate_gates'][method]['gate_evaluable']for v in datasets.values()for method in ('flow_pose_contact','mixture_pose_contact'))
    default={name:all(v['candidate_gates'][name]['dataset_gate_pass']for v in datasets.values())for name in ('flow_pose_contact','mixture_pose_contact')}
    out={'scope':'Eight historically exposed cases; frozen comparison, not unseen-object validation',
         'all_required_comparisons_evaluable':all_evaluable,'default_candidate_gate':default,'gate_interpretation':'Narrow frozen adoption criterion against nearest-bank pose/contact only; passing does not imply best overall prediction or improvement over velocity/persistence',
         'cases':cases,'datasets':datasets,'source_sha256':sources,
         'limitations':['Coordinate gains do not imply shape accuracy or physical identification','Pose/contact includes projected rigid correction, not coupled mechanics','No calibrated real parameter posterior or rendering gain established','Different initial-geometry-admitted node counts are matched within each case only','Nearest-bank adoption gate is narrow; velocity/persistence deficits are reported separately and may contradict a global superiority claim']}
    (REPORT/'compactexpansion-results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['FROZEN EIGHT-CASE DEVELOPMENT EXPANSION','Comparisons complete: '+str(all_evaluable),'']
    for row in cases:
        lines.append(row['case'])
        for name,m in row['methods'].items():lines.append('  '+name+': '+str(m['coordinate_rmse_mm'])+' mm coordinate; '+str(m['aligned_shape_rmse_mm'])+' mm shape; complete='+str(m['complete']))
    for dataset,result in datasets.items():
        lines.append(dataset+' strong-baseline deficits (positive=worse): '+json.dumps(result['strong_baseline_comparisons']))
    lines+=['','Narrow nearest-bank-only candidate gate (not global superiority): '+str(default),'Missing/failed required comparisons prevent promotion. Full conditional means and per-case differences: compactexpansion-results.json.']
    (REPORT/'compactexpansion-results.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cases_with_any_scores':sum(bool(r['methods'])for r in cases),'all_evaluable':all_evaluable,'candidate_gate':default}))

if __name__=='__main__':summarize()
