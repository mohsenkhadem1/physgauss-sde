"""Bounded development selection: EXACTLY best/last x flow/MDN, recorded VAL.

No training, candidate search, TEST observations/targets, or changes to the
original exposed-test study. All176 VAL predictions seal before VAL truth reads.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.gaussian_boundary_flow_v02.isaac_bridge_campaign import MODELS,validate_recipe,load_cache
from scripts.gaussian_boundary_flow_v02.isaac_prefix_cache import sha,write_json
from scripts.gaussian_boundary_flow_v02.isaac_recorded_evaluation import trajectory_metrics,parameter_metrics,validate_identity
from train_gaussian_inverse import load_frozen_operator
from evaluate_gaussian_inverse import predict_particles

ROOT=Path(__file__).resolve().parents[2]
CANDIDATES=('flow_best','flow_last','mdn_best','mdn_last')
SOURCE_FILES=[Path(__file__),ROOT/'scripts/gaussian_boundary_flow_v02/isaac_recorded_evaluation.py']

def source_hashes():return {str(p.resolve().relative_to(ROOT)):sha(p) for p in SOURCE_FILES}


def bound_validation_roster(manifest,catalog,test_seal):
    allowed={r['key']:r for r in catalog['cases'] if r['split']=='validation'}
    test_paths={str((Path(d['root'])/'episodes'/r['episode_id']).resolve()) for d in test_seal['datasets'] for r in d['cases']}
    test_groups={(str(Path(d['root']).resolve()),r['group_id']) for d in test_seal['datasets'] for r in d['cases']}
    rows=[]
    for row in manifest['cases']:
        if row['role']!='validation':continue
        ref=allowed.get(row['key'])
        if ref is None or (row['group'],row['recipe_sha256'],row['dataset'])!=(ref['group_id'],ref['recipe_sha256'],row['key'].rsplit('/',1)[0]):raise ValueError('Validation cache/catalog identity mismatch')
        episode=Path(ref['path']).resolve()
        if str(episode) in test_paths or (str(episode.parent.parent),row['group']) in test_groups:raise ValueError('Validation roster overlaps historical TEST identities/groups')
        rows.append(dict(key=row['key'],case=row['key'].replace('/','--'),dataset=row['dataset'],group=row['group'],cache=row['cache'],cache_sha256=row['cache_sha256'],episode=str(episode)))
    if len(rows)!=44 or len({r['key'] for r in rows})!=44 or set(allowed)!={r['key'] for r in rows}:raise ValueError('Exactly44 bound validation identities required')
    return rows


def freeze(output,training='runs/gaussian-boundary-isaac-bridge-v01'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'plan.json'
    if path.exists():raise FileExistsError('Selection plan immutable')
    training=Path(training).resolve();recipe=validate_recipe(training);seal_path=training/'training-sealed.json';seal=json.loads(seal_path.read_text())
    if seal['recipe_sha256']!=sha(training/'recipe.json'):raise ValueError('Training seal binding changed')
    load_cache(recipe['cache'])  # Existing exactcatalog/role/group/source/shard checks; no trajectory files.
    catalog_path=ROOT/'runs/gaussian-final-passive-catalog.json';catalog=json.loads(catalog_path.read_text())
    manifest_path=Path(recipe['cache'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    test_seal=json.loads(Path(recipe['exposed_test_seal']).read_text())
    rows=bound_validation_roster(manifest,catalog,test_seal)
    candidates={};hash_owner={}
    for kind in ('flow','mdn'):
        for snapshot in ('best','last'):
            name=f'{kind}_{snapshot}';checkpoint=training/kind/f'{snapshot}.pt';digest=sha(checkpoint)
            declared=seal['models'][kind]['artifact_sha256'].get(str(checkpoint.relative_to(ROOT)))
            if declared!=digest:raise ValueError('Candidate differs from root sealed checkpoint')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['kind']!=kind or state['recipe_sha256']!=seal['recipe_sha256'] or not 1<=state['step']<=2400:raise ValueError('Candidate checkpoint identity mismatch')
            duplicate=hash_owner.get(digest);hash_owner.setdefault(digest,name)
            candidates[name]=dict(kind=kind,snapshot=snapshot,checkpoint=str(checkpoint),checkpoint_sha256=digest,step=state['step'],duplicate_of=duplicate)
    plan=dict(version='bounded-recorded-validation-selection-v01',development=True,
        interpretation='Selection changed after seeing the first exposed TEST study. This is development, not fresh confirmation; no TEST rerun in this step.',
        training=str(training),training_recipe_sha256=sha(training/'recipe.json'),training_seal=str(seal_path),training_seal_sha256=sha(seal_path),
        cache=recipe['cache'],cache_manifest_sha256=sha(manifest_path),catalog=str(catalog_path),catalog_sha256=sha(catalog_path),
        exposed_test_seal=recipe['exposed_test_seal'],exposed_test_seal_sha256=recipe['exposed_test_seal_sha256'],source_sha256=source_hashes(),teacher=recipe['teacher'],teacher_sha256=recipe['teacher_sha256'],candidates=candidates,cases=rows,
        predictions=dict(samples=128,steps=32,seed=61053,clock_s=(np.arange(61)/20).tolist(),nodes=455,prefix_times_s=[0,.25,.5,.75,1.],feature_shape=[5,119],same_seed_for_checkpoints=True),
        selection=dict(primary='Equal-case mean recorded coordinate RMSE, frames40..60 (2–3s), all455nodes,128draw posterior mean',
                       required_complete_cases=44,tie_break='retain original best checkpoint for exact equal scalar score',
                       failure_rule='Candidate with any prediction/identity/evaluation failure is ineligible; no case replacement or silent exclusions',
                       additional_candidates=0,retraining=False),
        diagnostics=['rigid-aligned shape RMSE','pointwise90%intervalcoverage/width/score','jointtrajectory energy score','activeparameter/material/category metrics'],
        target_access_gate='Existing all44x4 terminal prediction barrier; validation-role whitelist checked before trajectory access')
    write_json(path,plan);return plan


def verify(output):
    output=Path(output);plan=json.loads((output/'plan.json').read_text())
    if plan['source_sha256']!=source_hashes():raise ValueError('Selection source changed')
    recipe=validate_recipe(plan['training'])
    bindings=[(Path(plan['training'])/'recipe.json',plan['training_recipe_sha256']),(plan['training_seal'],plan['training_seal_sha256']),
              (Path(plan['cache'])/'manifest.json',plan['cache_manifest_sha256']),(plan['catalog'],plan['catalog_sha256']),(plan['teacher'],plan['teacher_sha256']),(plan['exposed_test_seal'],plan['exposed_test_seal_sha256'])]
    if any(sha(path)!=digest for path,digest in bindings):raise ValueError('Bound selection artifact changed')
    if set(plan['candidates'])!=set(CANDIDATES) or len(plan['cases'])!=44 or len({r['key'] for r in plan['cases']})!=44:raise ValueError('Bounded candidate/case roster changed')
    expected=bound_validation_roster(json.loads((Path(plan['cache'])/'manifest.json').read_text()),json.loads(Path(plan['catalog']).read_text()),json.loads(Path(plan['exposed_test_seal']).read_text()))
    if plan['cases']!=expected:raise ValueError('Plan case/cache/dataset/path bindings differ from frozen validation roster')
    for name,candidate in plan['candidates'].items():
        if name!=candidate['kind']+'_'+candidate['snapshot'] or candidate['checkpoint']!=str(Path(plan['training'])/candidate['kind']/(candidate['snapshot']+'.pt')):raise ValueError('Candidate identity changed')
        if sha(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('Fixed candidate changed')
    budget=plan['predictions']
    if budget['samples']!=128 or budget['steps']!=32 or budget['nodes']!=455 or budget['clock_s']!=(np.arange(61)/20).tolist() or budget['prefix_times_s']!=[0,.25,.5,.75,1.] or budget['feature_shape']!=[5,119] or budget['seed']!=61053:raise ValueError('Frozen prediction budget changed')
    return plan


def terminal(output,row,name,plan):
    output=Path(output);run=output/row['case']/name;path=run/'prediction-status.json'
    if not path.exists():return None
    status=json.loads(path.read_text())
    if status['case']!=row['case'] or status['candidate']!=name or status['plan_sha256']!=sha(output/'plan.json'):raise ValueError('Prediction status binding mismatch')
    if status['state']=='failed':return status
    if status['state']!='complete':raise ValueError('Unknown prediction state')
    seal=json.loads((run/'prediction-seal.json').read_text())
    if status['seal_sha256']!=sha(run/'prediction-seal.json') or seal['checkpoint_sha256']!=plan['candidates'][name]['checkpoint_sha256'] or seal['cache_sha256']!=row['cache_sha256'] or seal['teacher_sha256']!=plan['teacher_sha256']:raise ValueError('Prediction seal binding mismatch')
    owner=plan['candidates'][name]['duplicate_of'] or name
    expected_path=(output/row['case']/owner/'prediction.npz').resolve()
    if Path(seal['prediction_path']).resolve()!=expected_path or seal['case']!=row['case'] or seal['candidate']!=name or seal['plan_sha256']!=sha(output/'plan.json'):raise ValueError('Prediction file identity mismatch')
    if seal['prediction_sha256']!=sha(expected_path):raise ValueError('Prediction bytes changed')
    return status


def predict(output,device='cuda:0'):
    output=Path(output);plan=verify(output);torch.set_num_threads(2);cache={r['key']:r for r in load_cache(plan['cache']) if r['role']=='validation'}
    forward=load_frozen_operator(plan['teacher'],device);models={}
    for name,candidate in plan['candidates'].items():
        if candidate['duplicate_of']:continue
        state=torch.load(candidate['checkpoint'],map_location=device,weights_only=False);model=MODELS[candidate['kind']]().to(device);model.load_state_dict(state['model']);model.eval();models[name]=model
    ph=sha(output/'plan.json')
    for index,row in enumerate(plan['cases']):
        arrays=cache[row['key']]
        for name in CANDIDATES:
            if terminal(output,row,name,plan) is not None:continue
            run=output/row['case']/name;run.mkdir(parents=True,exist_ok=True);candidate=plan['candidates'][name];status=dict(case=row['case'],candidate=name,plan_sha256=ph)
            try:
                if any(run.iterdir()):raise ValueError('Unfinished prediction artifacts preserved; no retry')
                if candidate['duplicate_of']:
                    owner=output/row['case']/candidate['duplicate_of'];ownerstatus=terminal(output,row,candidate['duplicate_of'],plan)
                    if ownerstatus is None or ownerstatus['state']!='complete':raise ValueError('Duplicate original did not complete')
                    original=json.loads((owner/'prediction-seal.json').read_text());prediction_path=Path(original['prediction_path'])
                else:
                    generator=torch.Generator(device=device).manual_seed(plan['predictions']['seed']+index)
                    features=torch.tensor(arrays['features'][None],device=device)
                    draw=models[name].sample(features,samples=128,steps=32,generator=generator);particles={k:v[0] for k,v in draw.items()}
                    positions=predict_particles(dict(device=device,forward=forward),particles,dict(rest_points_m=arrays['rest']),plan['predictions']['clock_s'])
                    if positions.shape!=(128,61,455,3) or not torch.isfinite(positions).all() or any(not torch.isfinite(p).all() for p in particles.values()):raise ValueError('Invalid forecast; no draw filtering')
                    values=dict(positions_m=positions.cpu().numpy(),mean_positions_m=positions.mean(0).cpu().numpy(),times_s=np.asarray(plan['predictions']['clock_s'],np.float32),rest_points_m=arrays['rest'],input_times_s=arrays['times'],input_frame_ids=arrays['frame_ids'])
                    values.update({k:v.cpu().numpy() for k,v in particles.items()});prediction_path=run/'prediction.npz';temp=run/'prediction.tmp.npz';np.savez_compressed(temp,**values);temp.replace(prediction_path)
                seal=dict(case=row['case'],candidate=name,plan_sha256=ph,checkpoint_sha256=candidate['checkpoint_sha256'],cache_sha256=row['cache_sha256'],teacher_sha256=plan['teacher_sha256'],prediction_path=str(prediction_path.resolve()),prediction_sha256=sha(prediction_path),seed=plan['predictions']['seed']+index,duplicate_of=candidate['duplicate_of'],no_trajectory_targets=True)
                write_json(run/'prediction-seal.json',seal);status.update(state='complete',seal_sha256=sha(run/'prediction-seal.json'))
            except Exception as exc:status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
            write_json(run/'prediction-status.json',status)
        write_json(output/'prediction-progress.json',dict(cases=index+1,expected=44,plan_sha256=ph))


def barrier(output):
    output=Path(output);plan=verify(output);rows=[]
    for row in plan['cases']:
        for name in CANDIDATES:
            status=terminal(output,row,name,plan)
            if status is None:raise ValueError('All176 validation predictions must finish before targets')
            rows.append(dict(case=row['case'],candidate=name,state=status['state'],status_sha256=sha(output/row['case']/name/'prediction-status.json')))
    result=dict(plan_sha256=sha(output/'plan.json'),terminal=176,complete=sum(r['state']=='complete' for r in rows),failed=sum(r['state']=='failed' for r in rows),records=rows)
    path=output/'barrier.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('Barrier changed')
    else:write_json(path,result)
    return result


def validation_truth(row,plan,arrays):
    # Recheck role before any target access. NEVER accept a caller-supplied TEST path.
    catalog=json.loads(Path(plan['catalog']).read_text());match=[r for r in catalog['cases'] if r['key']==row['key']]
    if len(match)!=1 or match[0]['split']!='validation' or Path(match[0]['path']).resolve()!=Path(row['episode']).resolve():raise ValueError('Only declared VALIDATION targets permitted')
    path=Path(row['episode'])/'truth/trajectory.npz'
    with np.load(path,allow_pickle=False) as f:target=f['positions_m'];times=f['times_s'];simulated=f['simulated_times_s']
    reference_path=Path(row['episode'])/'reference.npz'
    with np.load(reference_path,allow_pickle=False) as f:reference=f['rest_points_m']
    return target,times,simulated,reference,{str(path):sha(path),str(reference_path):sha(reference_path)}


def choose(rows):
    summaries={};choices={}
    metrics=['coordinate_rmse_m','rigid_aligned_shape_rmse_m','coordinate90_coverage','coordinate90_width_m','coordinate90_interval_score_m','trajectory_energy_score_m','active_parameter90_coverage','active_parameter90_normalized_width','active_parameter_normalized_absolute_error']
    for name in CANDIDATES:
        selected=[r for r in rows if r['candidate']==name];good=[r for r in selected if r['state']=='complete'];eligible=len(selected)==44 and len(good)==44 and len({r['case'] for r in good})==44
        summaries[name]=dict(expected=44,complete=len(good),failed=len(selected)-len(good),eligible=eligible,excluded_cases=[r['case'] for r in selected if r['state']!='complete'],episode_macro={key:float(np.mean([r['metrics'][key] for r in good])) if good else None for key in metrics})
    for kind in ('flow','mdn'):
        names=[f'{kind}_best',f'{kind}_last'];eligible=[n for n in names if summaries[n]['eligible']]
        selected=min(eligible,key=lambda n:(summaries[n]['episode_macro']['coordinate_rmse_m'],names.index(n))) if eligible else None
        choices[kind]=dict(selected=selected,changed_from_original=selected==f'{kind}_last',selection_basis='44-case recorded validation coordinate RMSE only',development=True)
    return dict(candidates=summaries,choices=choices,test_re_evaluation=False,qualification='Development checkpoint selection after first exposed-test study, no fresh confirmation claim')


def score(output):
    output=Path(output);plan=verify(output)
    if not(output/'barrier.json').exists():raise ValueError('Prediction barrier absent; no validation trajectory reads')
    barrier(output);binding=dict(plan_sha256=sha(output/'plan.json'),barrier_sha256=sha(output/'barrier.json'))
    cache={r['key']:r for r in load_cache(plan['cache']) if r['role']=='validation'};rows=[]
    for row in plan['cases']:
        arrays=cache[row['key']]
        # Existing fixed labels were already legal training/VAL supervision.
        try:
            target,times,simulated,reference,target_hashes=validation_truth(row,plan,arrays);target_error=None
        except Exception as exc:target_error=(type(exc).__name__,str(exc));target_hashes={}
        for name in CANDIDATES:
            run=output/row['case']/name;path=run/'score.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['binding']!=binding or old['case']!=row['case'] or old['candidate']!=name or old['target_sha256']!=target_hashes:raise ValueError('Existing validation score changed')
                rows.append(old);continue
            result=dict(case=row['case'],candidate=name,binding=binding,target_sha256=target_hashes);status=terminal(output,row,name,plan)
            if status['state']!='complete':result.update(state='prediction_failed',error=status.get('error'))
            else:
                try:
                    if target_error:raise ValueError('Recorded validation target read failed: '+repr(target_error))
                    seal=json.loads((run/'prediction-seal.json').read_text())
                    with np.load(seal['prediction_path'],allow_pickle=False) as f:pred={k:f[k] for k in f.files}
                    public=dict(rest=arrays['rest'],times=arrays['times'],frame_ids=arrays['frame_ids'])
                    validate_identity(pred,public,target,reference,times,simulated,128)
                    metrics=trajectory_metrics(pred['positions_m'][:,40:61],target[40:61],arrays['rest']);metrics.update(parameter_metrics(pred,arrays['physical'],int(arrays['boundary']),int(arrays['force'])))
                    result.update(state='complete',metrics=metrics,prediction_seal_sha256=sha(run/'prediction-seal.json'))
                except Exception as exc:result.update(state='evaluation_failed',error_type=type(exc).__name__,error=str(exc))
            write_json(path,result);rows.append(result)
        write_json(output/'scoring-progress.json',dict(cases=len(rows)//4,expected=44,binding=binding))
    summary=choose(rows);summary['binding']=binding;summary['score_sha256']={str(p.relative_to(output)):sha(p) for r in plan['cases'] for p in [output/r['case']/name/'score.json' for name in CANDIDATES]}
    path=output/'summary.json'
    if path.exists() and json.loads(path.read_text())!=summary:raise ValueError('Completed selection summary changed; no overwrite')
    if not path.exists():write_json(path,summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','predict','barrier','score']);p.add_argument('--output',default='runs/gaussian-boundary-isaac-selection-v01');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='freeze':freeze(a.output)
    elif a.command=='predict':predict(a.output,a.device)
    elif a.command=='barrier':barrier(a.output)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):score(a.output)
