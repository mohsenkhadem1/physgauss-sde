"""Additive, failure-retaining cohort image prediction seals. No target reader."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import matched_appearance_native_resolution as fit
from scripts.compositional_dynamics_v17.cohort_geometry_evaluation import committed
ROOT=fit.ROOT; sha=fit.sha;read=fit.read;write=fit.write
BASE=ROOT/'runs/compositional-dynamics-v17/fresh-cohort-appearance'
PLAN=ROOT/'reports/compositional-dynamics-v17/fresh-cohort-appearance-plan.json'

def prepare(bindings):
    if PLAN.exists()or BASE.exists():raise ValueError('Preserve previous image study')
    protocol,attempts,evidence=committed(bindings) # All16, including explicit failures; no references.
    fit.guard();appearance=read(fit.BASE/'training-complete.json')
    if not appearance['complete']or appearance['gaussian_count']<=0:raise ValueError('Valid appearance required')
    cp=ROOT/appearance['checkpoint']
    if sha(cp)!=appearance['checkpoint_sha256']:raise ValueError('Appearance changed')
    evidence[str(cp.relative_to(ROOT))]=sha(cp)
    for f in [fit.PLAN,fit.DESIGN,fit.BASE/'training-complete.json']:evidence[str(f.relative_to(ROOT))]=sha(f)
    queries=[]
    for world in protocol['generator_settings']['world_ids']:
        for role in ['forecast-40mm','forecast-80mm']:
            rows=[r for r in attempts if r['world_id']==world and r['role']==role]
            methods=[];failed={}
            for row in rows:
                method={'ours':'shared_q2','native_phystwin':'native'}[row['method']]
                if row['record']['completed']:
                    methods.append(method)
                    if method=='shared_q2':
                        f=row['array'].parent/'trajectory.npz'
                        if sha(f)!=row['record']['trajectory_sha256']:raise ValueError('Full Q2 trajectory changed')
                        evidence[str(f.relative_to(ROOT))]=sha(f)
                else:failed[method]=row['record'].get('error','Incomplete physical forecast')
            root='runs/compositional-dynamics-v17/fresh-cohort'
            queries.append(dict(world_id=world,role=role,methods=methods,failed=failed,native_directory=f'{root}/native/{world}/forecasts/{role}',native_array='prediction.npz',ours_directory=f'{root}/shared-forecast/{world}/{role}',q2_n=3,geometry_dimensions_m=protocol['generator_settings']['dimensions_m'],times_s=(np.arange(31)*.01).tolist(),cameras=read(fit.DESIGN)['cameras'],checkpoint=appearance['checkpoint']))
    from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
    sources=closure([Path(__file__),ROOT/'scripts/compositional_dynamics_v17/cohort_appearance_transport.py',ROOT/'scripts/compositional_dynamics_v17/cohort_appearance_score.py'])+list(fit.AUTHOR.rglob('*.py'))
    write(PLAN,dict(bindings=bindings,queries=queries,source_sha256={str(f.relative_to(ROOT)):sha(f)for f in sources},input_sha256=evidence,expected_queries=8,expected_image_slots=2480,scope='Four fresh worlds, two loads; same learned initial Gaussian appearance, native K16 versus Q2 F-covariance transport. Not full native reconstruction. All failures retained; finite-resolution references, no continuum qualification.',target_gate='All16 physical commits before preparation; all2480 image slots committed success/failure before any target rendering.'))

def guard():
    p=read(PLAN)
    for group in ['source_sha256','input_sha256']:
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Changed source/input '+name)
    committed(p['bindings'])
    return p

def predict():
    p=guard()
    if BASE.exists():raise ValueError('Preserve previous predictions')
    BASE.mkdir(parents=True);ledger=[]
    from scripts.compositional_dynamics_v17.cohort_appearance_transport import render_query
    for query in p['queries']:
        rendered={};error=None
        try:
            if query['methods']:
                result=render_query(query,BASE/query['world_id']/query['role'])
                rendered={(r['method'],r['frame'],r['camera']):r for r in result['records']}
        except Exception as exc:error=repr(exc)
        for method in ['native','shared_q2']:
            for frame,t in enumerate(query['times_s']):
                for camera in query['cameras']:
                    key=(method,frame,camera['name']);record=rendered.get(key)
                    ledger.append(dict(world_id=query['world_id'],query=query['role'],method=method,frame=frame,time_s=t,camera=camera['name'],role=camera['role'],completed=record is not None,prediction=record,error=None if record else query['failed'].get(method,error or 'Render unavailable')))
        write(BASE/'progress.json',dict(plan_sha256=sha(PLAN),rows=ledger))
    guard();write(BASE/'prediction-commit.json',dict(plan_sha256=sha(PLAN),expected=2480,rows=ledger,reference_images_read=False))

def prediction_gate():
    p=guard();c=read(BASE/'prediction-commit.json')
    if c['plan_sha256']!=sha(PLAN):raise ValueError('Image plan changed')
    expected={(q['world_id'],q['role'],m,i,k['name'])for q in p['queries']for m in ['native','shared_q2']for i in range(31)for k in q['cameras']}
    keys=[]
    for r in c['rows']:
        keys.append((r['world_id'],r['query'],r['method'],r['frame'],r['camera']))
        if type(r['completed'])is not bool or r['time_s']!=r['frame']*.01:raise ValueError('Invalid image slot')
        if r['completed']:
            a=r['prediction']
            if (a['method'],a['frame'],a['camera'],a['time_s'])!=(r['method'],r['frame'],r['camera'],r['time_s'])or sha(ROOT/a['path'])!=a['sha256']:raise ValueError('Image seal changed')
    if len(keys)!=2480 or len(set(keys))!=2480 or set(keys)!=expected:raise ValueError('All2480 unique image slots required before references')
    return p,c

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','predict']);a.add_argument('--bindings',type=Path);args=a.parse_args()
    prepare(read(args.bindings))if args.stage=='prepare'else predict()
