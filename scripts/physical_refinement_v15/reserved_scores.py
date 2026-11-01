"""Score frozen reserved forecasts only AFTER the entire cohort is committed.

Every policy, pair and budget snapshot is retained. Random baselines average
errors/scores of all pairs, not predictions. Intervals resample worlds, keeping
all four correlated noise repeats together. This is a finite-ensemble exploratory
study; numerical mismatch and coverage failures are reported, not filtered.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.physical_refinement_v15.runtime_contract import verify
from scripts.physical_refinement_v15.banks import ROOT,checked_npz,nonrigid,sha
from scripts.physical_refinement_v15.quadrature import geometry_cases,hypotheses
from scripts.physical_refinement_v15.reserved_forecasts import schedules
from scripts.physical_refinement_v15.query_design import menu
from scripts.physical_refinement_v15.predictive_scores import marginal_scores
from scripts.phystwin.sequential_fixture import points
BASE=ROOT/'runs/physical-refinement-v15'


def committed(folder,protocol,protocol_sha,freeze_sha):
    """Verify every expected forecast before any response reader is called."""
    path=folder/'complete.json';record=json.loads(path.read_text())
    if record.get('role')!='all_forecasts_committed_before_response_scoring' or record.get('protocol_sha256')!=protocol_sha or record.get('method_freeze_sha256')!=freeze_sha:
        raise ValueError('Full frozen forecast commitment required before scoring')
    if not record['rows']:raise ValueError('Missing declared forecast cohort')
    generation_path=folder.parent/'measurements/complete.json'
    generation=json.loads(generation_path.read_text())
    if sha(generation_path)!=record['measurement_generation_sha256'] or generation['protocol_sha256']!=protocol_sha or generation['method_freeze_sha256']!=freeze_sha:raise ValueError('Measurement generation mismatch')
    measurement_hashes={r['path']:r['sha256'] for w in generation['worlds'] for r in w['manifests']}
    expected_h=len(hypotheses(protocol['gradient_count'],protocol['scale_count'])[0])
    indexed={r['path']:r['sha256'] for r in record['rows']}
    if len(indexed)!=len(record['rows']):raise ValueError('Duplicate committed forecast')
    expected={};results=[];evidence={str(path.relative_to(ROOT)):sha(path)} if path.is_relative_to(ROOT) else {str(path):sha(path)}
    for world in protocol['reserved_worlds']:
        for repeat in range(protocol['noise_repeats']):
            for track in protocol['tracks']:
                queries=[SimpleNamespace(name=x[0]) for x in menu(world['initial_camera'])]
                for schedule in schedules(protocol,world['initial_camera'],track,queries):
                    relative=f"{world['name']}/repeat-{repeat}/{track}/{schedule['name']}";expected[relative]=True
                    if relative not in indexed:raise ValueError('Missing declared forecast: '+relative)
                    p=folder/relative/'complete.json'
                    if sha(p)!=indexed[relative]:raise ValueError('Changed forecast record')
                    meta=json.loads(p.read_text())
                    if (meta['world'],meta['initial_camera'],meta['repeat'],meta['track'],meta['policy'])!=(world['name'],world['initial_camera'],repeat,track,schedule['name']):raise ValueError('Forecast identity mismatch')
                    sequence=meta['rollout']['sequence'];budget=schedule['budget']
                    measurement_relative=f"{world['name']}/repeat-{repeat}/{track}/manifest.json"
                    measurement_path=folder.parent/'measurements'/measurement_relative
                    if sha(measurement_path)!=measurement_hashes[measurement_relative] or meta['measurement_manifest_sha256']!=measurement_hashes[measurement_relative]:raise ValueError('Measurement manifest mismatch')
                    measurement=json.loads(measurement_path.read_text());receipt=meta['measurement_receipt']
                    if receipt['budget']!=budget or receipt['consumed']!=['initial']+sequence or [x['name'] for x in receipt['packets']]!=receipt['consumed']:raise ValueError('Consumed measurement identity mismatch')
                    for entry in receipt['packets']:
                        declared=measurement['packets'][entry['name']];packet=folder.parent/'measurements'/declared['path']
                        if entry['sha256']!=declared['sha256'] or entry['dimension']!=declared['dimension'] or sha(packet)!=declared['sha256']:raise ValueError('Consumed packet changed')
                        evidence[str(packet.relative_to(ROOT))]=sha(packet)
                    evidence[str(measurement_path.relative_to(ROOT))]=sha(measurement_path)
                    evidence[str(generation_path.relative_to(ROOT))]=sha(generation_path)
                    if meta['rollout']['budget']!=budget or len(sequence)!=budget or len(set(sequence))!=budget or any(x not in [q.name for q in queries] for x in sequence):raise ValueError('Invalid acquisition sequence')
                    if schedule['policy']=='fixed' and sequence!=schedule['sequence']:raise ValueError('Changed fixed schedule')
                    weightpath=p.parent/'weights.npz'
                    if sha(weightpath)!=meta['weights_sha256']:raise ValueError('Changed posterior snapshot')
                    with np.load(weightpath,allow_pickle=False) as data:weights=data['weights'].copy()
                    if weights.ndim!=2 or weights.shape!=(budget+1,expected_h) or not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1):raise ValueError('Invalid committed weights')
                    results.append((meta,weights))
                    for item in (p,weightpath):evidence[str(item.relative_to(ROOT)) if item.is_relative_to(ROOT) else str(item)]=sha(item)
    if set(expected)!=set(indexed):raise ValueError('Forecast cohort mismatch')
    return results,evidence


def load_responses(folder,pose_indices):
    fields=[];wrenches=[];evidence={}
    from scripts.physical_refinement_v15.response_actions import response_actions
    expected_poses=list(response_actions().values())
    complete=json.loads((folder/'complete.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    if complete['state']!='complete' or complete['signature']!=manifest['signature']:raise ValueError('Uncommitted mechanics job')
    for name in ('complete.json','manifest.json'):evidence[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for j,i in enumerate(pose_indices):
        p=folder/f'pose-{i:03d}'
        record=json.loads((p/'complete.json').read_text());pose=expected_poses[j]
        if sha(p/'complete.json')!=complete['pose_completion_sha256'][str(i)] or record['signature']!=complete['signature'] or record['pose_index']!=i or not np.array_equal(record['pose_m_rad'],pose) or not np.array_equal(manifest['spec']['poses_m_rad'][i],pose):raise ValueError('Response pose identity mismatch')
        data,h=checked_npz(p,'private-state.npz');evidence.update(h)
        with data:fields.append(data['field'].copy())
        data,h=checked_npz(p,'reaction-wrench.npz');evidence.update(h)
        with data:
            origins=np.array([[0.,0.,0.],[.24,0.,0.]])+pose[:,:3]
            if not np.array_equal(data['origins_world_m'],origins):raise ValueError('Reaction wrench origin mismatch')
            wrenches.append(np.r_[data['forces_N'][1],data['torques_Nm'][1]])
    fields=np.asarray(fields);wrenches=np.asarray(wrenches)
    if fields.shape!=(4,81,3) or wrenches.shape!=(4,6) or not np.isfinite(fields).all() or not np.isfinite(wrenches).all():raise ValueError('Finite four-action response fields/wrenches required')
    shapes=np.asarray([nonrigid(points(),x) for x in fields]).reshape(4,243)
    return fields,shapes,wrenches,evidence


def block_scales(shape_features=243):
    n=shape_features+6
    return np.r_[np.full(shape_features,.001*np.sqrt(shape_features/(.5*n))),
                 np.full(3,.1*np.sqrt(3/(.25*n))),np.full(3,.01*np.sqrt(3/(.25*n)))]


def point_metrics(field,wrench,target_field,target_wrench,canonical_shape,target_shape):
    residual=[];rest=points()
    for a,b in zip(rest+field,rest+target_field):
        a=a-a.mean(0);b=b-b.mean(0);u,_,vt=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt)
        residual.append(np.mean((a@(u@fix@vt)-b)**2))
    design=np.concatenate([canonical_shape,wrench],axis=-1);target=np.concatenate([target_shape,target_wrench],axis=-1)
    loss=float(np.mean(((design-target)/block_scales())**2))
    return dict(coordinate_rmse_mm=float(1000*np.sqrt(np.mean((field-target_field)**2))),
                aligned_shape_rmse_mm=float(1000*np.sqrt(np.mean(residual))),
                reaction_force_rmse_N=float(np.sqrt(np.mean((wrench[:,:3]-target_wrench[:,:3])**2))),
                reaction_torque_rmse_Nm=float(np.sqrt(np.mean((wrench[:,3:]-target_wrench[:,3:])**2))),
                normalized_design_mse=loss,normalized_design_utility=-loss)


def score_weights(weights,fields,shapes,wrenches,indices,scales,target):
    if weights.shape!=(len(indices),):raise ValueError('Weights/hypothesis dimensions differ')
    geom=np.bincount(indices,weights=weights,minlength=len(fields));field=np.einsum('h,h...->...',geom,fields)
    shape=np.einsum('h,h...->...',geom,shapes);expanded_wrench=wrenches[indices]*scales[:,None,None]
    wrench=np.einsum('h,h...->...',weights,expanded_wrench);tf,ts,tw=target
    mean=point_metrics(field,wrench,tf,tw,shape,ts);mode=int(np.argmax(weights));bi=indices[mode]
    highest=point_metrics(fields[bi],expanded_wrench[mode],tf,tw,shapes[bi],ts)
    proper={}
    for label,values,w,t in [('shape_m',shapes,geom,ts),('force_N',expanded_wrench[:,:,:3],weights,tw[:,:3]),('torque_Nm',expanded_wrench[:,:,3:],weights,tw[:,3:])]:
        result=marginal_scores(values,w,t)
        for key in ('crps_mean','central90_coordinate_coverage','mean_interval_width'):proper[label+'_'+key]=result[key]
    return dict(metrics=mean,highest_weight_metrics=highest,highest_weight_index=mode,proper_scores=proper)


def averaged(rows):
    return {group:{key:float(np.mean([r[group][key] for r in rows])) for key in rows[0][group]}
            for group in ('metrics','highest_weight_metrics','proper_scores')}


def summaries(rows,bootstrap):
    final=[r for r in rows if r['is_final'] and not r['policy'].startswith(('pair-','single-'))]
    output=[]
    for camera in ('all','left','oblique'):
        for track in ('V','VF'):
            for policy in dict.fromkeys(r['policy'] for r in final):
                selected=[r for r in final if r['track']==track and r['policy']==policy and (camera=='all' or r['initial_camera']==camera)]
                if selected:output.append(dict(initial_camera=camera,track=track,policy=policy,count=len(selected),worlds=len({r['world'] for r in selected}),**averaged(selected)))
    paired=[];rng=np.random.default_rng(bootstrap['seed'])
    for track in ('V','VF'):
        active=[r for r in final if r['track']==track and r['policy']=='physical_risk']
        for baseline in ('initial_only','visual_uncertainty','fixed_all','fixed_mixed','fixed_views','fixed_probes','random_exact','random_probes'):
            others={(r['world'],r['repeat']):r for r in final if r['track']==track and r['policy']==baseline}
            if not others:continue
            worlds=sorted({r['world'] for r in active});draws=rng.integers(len(worlds),size=(bootstrap['replicates'],len(worlds)))
            results={}
            for key in active[0]['metrics']:
                means=np.array([np.mean([r['metrics'][key]-others[(r['world'],r['repeat'])]['metrics'][key] for r in active if r['world']==world]) for world in worlds])
                boot=means[draws].mean(1);results[key]=dict(mean_active_minus_baseline=float(means.mean()),world_bootstrap95=np.quantile(boot,bootstrap['interval']).tolist(),world_differences=means.tolist())
            paired.append(dict(track=track,baseline=baseline,world_names=worlds,world_count=len(worlds),bootstrap_draws=bootstrap['replicates'],metrics=results))
    return output,paired


def run(protocol_path,protocol_sha,freeze_path,freeze_sha):
    protocol,freeze=verify(protocol_path,protocol_sha,freeze_path,freeze_sha,artifacts=True)
    out=ROOT/'reports/physical-refinement-v15/reserved-scores.json'
    if out.exists():raise ValueError('Preserve existing reserved scores')
    forecasts,evidence=committed(BASE/'reserved/forecasts',protocol,protocol_sha,freeze_sha)
    qualification_path=ROOT/'reports/physical-refinement-v15/reserved-mechanics-qualification.json'
    qualification=json.loads(qualification_path.read_text())
    if qualification['protocol_sha256']!=protocol_sha or qualification['freeze_sha256']!=freeze_sha:raise ValueError('Qualification protocol mismatch')
    expected_jobs={(w['name'],n) for w in protocol['reserved_worlds'] for n in (4,6)}
    if {(j['job']['name'],j['job']['n']) for j in qualification['jobs']}!=expected_jobs or len(qualification['jobs'])!=24:raise ValueError('All24 truth jobs must be accounted for')
    if {c['world'] for c in qualification['comparisons']}!={w['name'] for w in protocol['reserved_worlds']}:raise ValueError('Missing truth qualifications')
    evidence[str(qualification_path.relative_to(ROOT))]=sha(qualification_path)
    # FIRST target/predicted response accesses occur strictly after complete cohort validation.
    cases=geometry_cases(protocol['gradient_count']);hypotheses_rows,_=hypotheses(protocol['gradient_count'],protocol['scale_count'])
    names=[c['name'] for c in cases];indices=np.array([names.index(r['base_name']) for r in hypotheses_rows]);scales=np.array([r['scale'] for r in hypotheses_rows])
    fields=[];shapes=[];wrenches=[]
    for name in names:
        f,s,w,h=load_responses(BASE/'response-predictions/hypotheses'/name/'n4',range(1,5));fields.append(f);shapes.append(s);wrenches.append(w);evidence.update(h)
    fields,shapes,wrenches=map(np.asarray,(fields,shapes,wrenches));rows=[]
    for world in protocol['reserved_worlds']:
        tf,ts,tw,h=load_responses(BASE/'reserved/worlds'/world['name']/'n6',range(5,9));evidence.update(h);cache={}
        for meta,weights in forecasts:
            if meta['world']!=world['name']:continue
            for budget,weight in enumerate(weights):
                key=hashlib.sha256(weight.tobytes()).hexdigest()
                if key not in cache:cache[key]=score_weights(weight,fields,shapes,wrenches,indices,scales,(tf,ts,tw))
                rows.append(dict(world=world['name'],initial_camera=world['initial_camera'],track=meta['track'],repeat=meta['repeat'],policy=meta['policy'],budget=budget,
                    is_final=budget==len(weights)-1,sequence=meta['rollout']['sequence'][:budget],acquisition=meta['rollout']['acquisition'][:budget],weights_sha256=key,**cache[key]))
        print(world['name'],'all responses scored',flush=True)
    for world in protocol['reserved_worlds']:
        for track in protocol['tracks']:
            for repeat in range(protocol['noise_repeats']):
                pairs=[r for r in rows if r['world']==world['name'] and r['track']==track and r['repeat']==repeat and r['policy'].startswith('pair-') and r['is_final']]
                if len(pairs)!=15:raise ValueError('All fifteen pair outcomes required')
                for name,selected in [('random_exact',pairs),('random_probes',[r for r in pairs if all(not n.startswith('view_') for n in r['sequence'])])]:
                    if len(selected)!=(15 if name=='random_exact' else 6):raise ValueError('Complete pair expectation required')
                    rows.append(dict(world=world['name'],initial_camera=world['initial_camera'],track=track,repeat=repeat,policy=name,budget=2,is_final=True,sequence=None,pair_count=len(selected),**averaged(selected)))
    for world in protocol['reserved_worlds']:
        for track in protocol['tracks']:
            chosen=next(x['query'] for x in protocol['fixed_budget_one'] if x['initial_camera']==world['initial_camera'] and x['track']==track)
            for repeat in range(protocol['noise_repeats']):
                singles=[r for r in rows if r['world']==world['name'] and r['track']==track and r['repeat']==repeat and r['policy'].startswith('single-') and r['is_final']]
                if len(singles)!=6:raise ValueError('All six budget-one outcomes required')
                selected=next(r for r in singles if r['sequence']==[chosen])
                rows.append({**selected,'policy':'fixed_budget_one','control_scope':'Separately development-trained single query, not first step of best pair'})
                rows.append(dict(world=world['name'],initial_camera=world['initial_camera'],track=track,repeat=repeat,policy='random_budget_one',budget=1,is_final=True,sequence=None,query_count=6,**averaged(singles)))
    risk_transfer=[]
    for row in rows:
        if row['policy']=='physical_risk' and row['budget']>0:
            previous=next(r for r in rows if r['world']==row['world'] and r['track']==row['track'] and r['repeat']==row['repeat'] and r['policy']==row['policy'] and r['budget']==row['budget']-1)
            chosen=row['sequence'][-1];acq=row['acquisition'][-1]
            risk_transfer.append(dict(world=row['world'],track=row['track'],repeat=row['repeat'],budget=row['budget'],query=chosen,
                estimated_development_design_gain=acq['estimated_scores'][chosen],mcse=acq['monte_carlo_diagnostics'][chosen]['standard_error'],
                realized_held_response_gain=previous['metrics']['normalized_design_mse']-row['metrics']['normalized_design_mse'],
                caveat='Acquisition uses three development actions; realized score uses four held response actions'))
    summary,paired=summaries(rows,protocol['bootstrap'])
    verify(protocol_path,protocol_sha,freeze_path,freeze_sha,artifacts=True)
    if any(sha(ROOT/p)!=h for p,h in evidence.items()):raise ValueError('Scored input changed')
    result=dict(scope=__doc__,protocol_sha256=protocol_sha,method_freeze_sha256=freeze_sha,source_sha256=freeze['source_sha256'],
                evidence_sha256=evidence,mechanical_qualification=qualification,rows=rows,summary=summary,paired_world_bootstrap=paired,estimated_vs_realized=risk_transfer,
                notes=['Highest-weight control is a quadrature-node MAP, dependent on prior quadrature density; not continuous-density MAP.',
                       'All failed numerical qualification flags must accompany scores; no qualified-case filtering.',
                       '90% intervals are marginal predictive coordinate coverage, not calibrated simultaneous confidence.',
                       'Bootstrap resamples twelve worlds after averaging four repeats; descriptive exploratory uncertainty.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args();run(a.protocol,a.protocol_sha256,a.freeze,a.freeze_sha256)
