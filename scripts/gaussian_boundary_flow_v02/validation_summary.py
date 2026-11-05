"""Failure-aware summary of already evaluated, sealed validation predictions.

No HDF5/package trajectory reads and no evaluator calls. Secondary rigid-shape
ablation is separately labelled and does not alter primary adoption criteria.
"""
import argparse,json,math
from pathlib import Path
from scripts.gaussian_boundary_flow_v02.object_score import load_sealed,sha

METHODS={
 'flow_raw':('physics','flow_raw'),'mixture_raw':('physics','mixture_raw'),
 'flow_tracking':('physics','flow_tracking'),'mixture_tracking':('physics','mixture_tracking'),
 'flow_pose_contact':('learned_contact','flow_pose'),'mixture_pose_contact':('learned_contact','mixture_pose'),
 'nearest_bank_raw':('retrieval','nearest_raw'),'nearest_bank_pose_contact':('retrieval_contact','nearest_contact'),
 'prefix_persistence':('baseline','prefix_persistence'),'prefix_velocity':('baseline','prefix_velocity'),
 'rigid_contact':('rigid_contact','rigid_contact')}
METRICS=('coordinate_rmse_mm','aligned_shape_rmse_mm')
SECONDARY='rigid_contact'


def verify_score(run,score,*,case,protocol_sha256,expected_budget=None):
    """Verify score identity/provenance against actual prediction seals, no targets."""
    run=Path(run);plan,seal,complete,pred=load_sealed(run)
    if plan['case']!=case or score.get('case')!=case:raise ValueError('Score/run case mismatch')
    if score.get('data_role')!='prospective_validation' or score.get('validation_protocol_sha256')!=protocol_sha256:
        raise ValueError('Score validation role/protocol mismatch')
    if plan.get('validation_protocol_sha256')!=protocol_sha256:raise ValueError('Run protocol mismatch')
    for key,path in [('prediction',run/'prediction.npz'),('plan',run/'plan.json')]:
        if score.get('source_sha256',{}).get(key)!=sha(path):raise ValueError('Score source digest mismatch: '+key)
    if score.get('interval')!=[complete['fit_end'],complete['frames']]:raise ValueError('Score interval differs from sealed full clock')
    expected_frames=complete['frames']-complete['fit_end']
    for item in score.get('methods',{}).values():
        counts=item['summary']
        if counts['total_frames']!=expected_frames or sum(counts[k] for k in ('valid_frames','failed_frames','no_reference_frames'))!=expected_frames:
            raise ValueError('Score frame accounting differs from sealed interval')
    budget={'frames':plan['prefix_frames'],'point_ids':plan['point_ids']}
    if score.get('baseline_budget')!=budget or (expected_budget is not None and budget!=expected_budget):
        raise ValueError('Score observation budget mismatch')
    if score.get('nodes')!=plan['args']['nodes'] or len(pred['nodes'])!=plan['args']['nodes']:
        raise ValueError('Score geometry budget mismatch')
    if score.get('observation_count')!=len(pred['observed_initial']) or len(score.get('observed_node_ids',[]))!=len(pred['observed_initial']):
        raise ValueError('Score observation mapping size mismatch')
    package=Path(plan['input_package'])
    if sha(package/'manifest.json')!=plan['input_package_manifest_sha256']:
        raise ValueError('Changed input manifest')
    return budget,(score['interval'],score['observed_node_ids'],plan['input_digest'])


def _mean(values):return sum(values)/len(values) if values else None


def aggregate_cases(rows):
    """Keep fixed denominators. Means and pairs are explicitly conditional."""
    cells=sorted({r['cell'] for r in rows});result={}
    for method in METHODS:
        good=[r for r in rows if r['methods'].get(method,{}).get('complete')]
        percell={}
        for cell in cells:
            selected=[r for r in rows if r['cell']==cell]
            valid=[r for r in good if r['cell']==cell]
            percell[cell]=dict(expected_cases=len(selected),complete_cases=len(valid),
                              conditional_means={m:_mean([r['methods'][method][m] for r in valid]) for m in METRICS})
        result[method]=dict(role='secondary' if method==SECONDARY else 'primary',expected_cases=len(rows),
             complete_cases=len(good),incomplete_cases=[r['case'] for r in rows if r not in good],
             conditional_equal_case_means={m:_mean([r['methods'][method][m] for r in good]) for m in METRICS},
             conditional_equal_cell_macro_means={m:_mean([c['conditional_means'][m] for c in percell.values() if c['conditional_means'][m] is not None]) for m in METRICS},
             expected_cells=len(cells),complete_cells=sum(c['complete_cases']==c['expected_cases'] for c in percell.values()),per_cell=percell)
    pairs={}
    for candidate in METHODS:
        for comparator in ['nearest_bank_pose_contact','rigid_contact','prefix_velocity','prefix_persistence']:
            if candidate==comparator:continue
            included=[r for r in rows if all(r['methods'].get(k,{}).get('complete') for k in [candidate,comparator])]
            key=candidate+' minus '+comparator
            pairs[key]=dict(role='secondary' if SECONDARY in (candidate,comparator) else 'primary',
                expected_cases=len(rows),paired_complete_cases=len(included),
                excluded_cases=[r['case'] for r in rows if r not in included],
                differences={m:[{'case':r['case'],'difference':r['methods'][candidate][m]-r['methods'][comparator][m]} for r in included] for m in METRICS},
                conditional_mean_difference={m:_mean([r['methods'][candidate][m]-r['methods'][comparator][m] for r in included]) for m in METRICS},
                wins={m:sum(r['methods'][candidate][m]<r['methods'][comparator][m] for r in included) for m in METRICS},
                sign='Negative favors candidate; no failed case imputed as zero')
    return result,pairs


def summarize(root,protocol_path):
    root=Path(root);protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text());digest=sha(protocol_path)
    if protocol.get('status')!='frozen' or protocol.get('role')!='validation' or len(protocol.get('cases',[]))!=12:
        raise ValueError('Frozen twelve-case validation protocol required')
    if len({r['case'] for r in protocol['cases']})!=12:raise ValueError('Duplicate validation case')
    for source,source_digest in protocol.get('source_sha256',{}).items():
        if sha(source)!=source_digest:raise ValueError('Changed frozen source: '+source)
    gatepath=root/'evaluation-gate.json'
    if not gatepath.exists():raise ValueError('Cohort evaluation barrier has not been sealed')
    gate=json.loads(gatepath.read_text())
    if gate.get('protocol_sha256')!=digest or gate.get('all12_terminal_before_any_target_read')is not True:
        raise ValueError('Cohort evaluation barrier mismatch')
    gate_status={r['case']:r['status_sha256'] for r in gate['cases']}
    if set(gate_status)!={r['case'] for r in protocol['cases']}:raise ValueError('Evaluation barrier cohort mismatch')
    rows=[];hashes={str(protocol_path):digest,str(gatepath):sha(gatepath)}
    for spec in protocol['cases']:
        case=spec['case'];folder=root/case;statepath=folder/'prediction-status.json'
        if not statepath.exists() or sha(statepath)!=gate_status[case]:raise ValueError('Primary terminal status differs from evaluation barrier')
        state=json.loads(statepath.read_text())
        if statepath.exists() and (state.get('case')!=case or state.get('protocol_sha256')!=digest):raise ValueError('Case status identity changed')
        row=dict(case=case,cell=f"{spec['subset']}/task{spec['task']}",prediction_status=state,methods={})
        documents={};identity=None;budget=None
        for stage in dict.fromkeys(stage for stage,_ in METHODS.values()):
            path=folder/(stage+'-scores.json');statuspath=folder/(stage+'-evaluation-status.json')
            status=json.loads(statuspath.read_text()) if statuspath.exists() else {'state':'missing'}
            if path.exists():
                if status.get('state')!='complete':raise ValueError('Score lacks completed evaluation status: '+str(path))
                if status.get('cohort_gate_sha256')!=sha(gatepath):raise ValueError('Score evaluation status has different cohort barrier')
                score=json.loads(path.read_text())
                if stage=='rigid_contact':
                    derived_plan=json.loads((folder/stage/'plan.json').read_text())
                    addendum=protocol_path.parent/'rigid-contact-secondary-addendum.json'
                    if derived_plan.get('secondary_addendum_sha256')!=sha(addendum):raise ValueError('Changed secondary addendum binding')
                    hashes[str(addendum)]=sha(addendum)
                    importpath=folder/'rigid_contact-import.json'
                    provenance=json.loads(importpath.read_text())
                    if provenance.get('secondary_addendum_sha256')!=sha(addendum):raise ValueError('Secondary import addendum mismatch')
                    for relative,file_digest in provenance['files_sha256'].items():
                        if sha(folder/stage/relative)!=file_digest:raise ValueError('Secondary import bytes changed')
                    hashes[str(importpath)]=sha(importpath)
                newbudget,newidentity=verify_score(folder/stage,score,case=case,protocol_sha256=digest,expected_budget=budget)
                if identity is not None and newidentity!=identity:raise ValueError('Cross-method target interval/IDs/input digest differ')
                budget,identity=newbudget,newidentity;documents[stage]=score;hashes[str(path)]=sha(path)
            elif status.get('state')=='complete':raise ValueError('Completed evaluation missing scores: '+str(path))
            for method,(owner,name) in METHODS.items():
                if owner!=stage:continue
                if stage not in documents:
                    row['methods'][method]=dict(complete=False,role='secondary' if method==SECONDARY else 'primary',reason='No verified completed score',evaluation_status=status,prediction_stage=state.get('stages',{}).get(stage));continue
                if name not in documents[stage]['methods']:raise ValueError('Missing declared method score: '+name)
                s=documents[stage]['methods'][name]['summary'];full=(s['valid_frames']==s['total_frames'] and s['failed_frames']==0 and s['no_reference_frames']==0)
                valid=all(isinstance(s.get(m),(int,float)) and math.isfinite(s[m]) for m in METRICS)
                row['methods'][method]=dict(complete=bool(full and valid),role='secondary' if method==SECONDARY else 'primary',
                                            **{k:s.get(k) for k in (*METRICS,'total_frames','valid_frames','failed_frames','no_reference_frames')})
        rows.append(row)
    methods,pairs=aggregate_cases(rows)
    return dict(scope='Twelve recurring-asset validation episodes; secondary rigid ablation labelled separately',
                cases=rows,methods=methods,paired_comparisons=pairs,source_sha256=hashes,
                caveats=['Conditional means retain explicit failed/missing denominators; no automatic superiority claim',
                         'Six cells contain two episodes each, not twelve independent objects','No pooled-frame significance',
                         'Rigid-removed shape error is distinct from rigid transport accuracy','Secondary ablation does not change frozen primary adoption gates'],
                target_data_access=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(a.root,a.protocol);out=Path(a.output)
    if out.exists():raise FileExistsError('Do not overwrite an existing summary')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
