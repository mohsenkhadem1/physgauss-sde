"""Inference-context/routing repair only; original render/scoring sources unchanged."""
import argparse,importlib.util,json,os
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import matched_appearance_render as original
from scripts.compositional_dynamics_v17_reporting.matched_video import five_forecasts
ROOT=original.ROOT;sha=original.sha;read=original.read;write=original.write
PLAN=ROOT/'reports/compositional-dynamics-v17/appearance-inference-recovery-plan.json'
OUT=ROOT/'runs/compositional-dynamics-v17/matched-appearance-inference-recovery'
REPORT=ROOT/'reports/compositional-dynamics-v17/matched-appearance-inference-recovery'

def clone(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare():
    if PLAN.exists()or OUT.exists():raise ValueError('Preserve recovery')
    p=original.guard();evidence=five_forecasts()
    if (original.BASE/'prediction-commit.json').exists()or list(original.BASE.rglob('*.npz')):raise ValueError('Expected original failure before image outputs')
    old={str(f.relative_to(ROOT)):sha(f)for f in original.BASE.rglob('*')if f.is_file()}
    sources=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/cohort_appearance_transport.py',ROOT/'scripts/compositional_dynamics_v17/matched_appearance_score.py',ROOT/'scripts/compositional_dynamics_v17_reporting/matched_video.py']
    write(PLAN,dict(scope='Same frozen renderer/physics/checkpoint/cameras/metrics; outer torch.no_grad covers opacity filtering so deepcopy operates on leaf tensors. New output routing, no scientific setting changes.',original_plan=str(original.PLAN.relative_to(ROOT)),original_plan_sha256=sha(original.PLAN),original_failed_directory=str(original.BASE.relative_to(ROOT)),original_failed_files=old,source_sha256={str(f.relative_to(ROOT)):sha(f)for f in sources},input_sha256=evidence,output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),smoke='Actual initial native and Q2 images for all5 cameras; same <=1e-4 RGB parity gate; finite positive Gaussians. No reference images.',cohort_scope='The same outer no_grad repair is required before first cohort rendering. Its routes need a separate explicit amendment; no cohort execution authorized by this development plan.'))

def guard():
    p=read(PLAN);original.guard();five_forecasts()
    if sha(ROOT/p['original_plan'])!=p['original_plan_sha256']:raise ValueError('Original plan changed')
    for group in ['source_sha256','input_sha256','original_failed_files']:
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Recovery evidence changed '+name)
    return p

def smoke():
    guard();p=original.guard();base=OUT/'initial-smoke';base.mkdir(parents=True,exist_ok=False)
    # Crop only already-committed prediction states at t0; no response/truth reads.
    native=base/'native';ours=base/'ours';native.mkdir();ours.mkdir()
    with np.load(ROOT/p['native_directory']/p['native_array'])as f:np.savez(native/'initial.npz',positions_m=f['positions_m'][:1],times_s=f['times_s'][:1])
    with np.load(ROOT/p['ours_directory']/'trajectory.npz')as f:np.savez(ours/'trajectory.npz',u=f['u'][:1],times_s=f['times_s'][:1],rest_nodes=f['rest_nodes'])
    q=dict(p,native_directory=str(native.relative_to(ROOT)),native_array='initial.npz',ours_directory=str(ours.relative_to(ROOT)),times_s=[0.])
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import torch
    from scripts.compositional_dynamics_v17.cohort_appearance_transport import render_query
    with torch.no_grad():result=render_query(q,base/'render')
    if len(result['records'])!=10 or result['gaussians']<=0 or result['initial_rgb_max_difference']>1e-4:raise ValueError('Actual initial render failed')
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(848*2,480*5),'white')
    for i,c in enumerate(p['cameras']):
        for j,m in enumerate(['native','shared_q2']):
            path=base/'render/predictions'/m/f"000-{c['name']}.npz"
            with np.load(path)as f:rgb=f['rgb']
            image=Image.fromarray(np.uint8(np.round(np.clip(rgb,0,1)*255)));sheet.paste(image,(j*848,i*480))
            ImageDraw.Draw(sheet).text((j*848+8,i*480+8),m+' '+c['name'],fill='black')
    REPORT.mkdir(parents=True,exist_ok=True);preview=REPORT/'initial-parity.png';sheet.save(preview)
    guard();write(REPORT/'initial-smoke.json',dict(plan_sha256=sha(PLAN),passed=True,**result,preview_sha256=sha(preview),reference_access=False))

def predictor():
    guard();s=read(REPORT/'initial-smoke.json')
    if not s['passed']or s['plan_sha256']!=sha(PLAN):raise ValueError('Actual render smoke required')
    for row in s['records']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Initial image changed')
    m=clone(Path(original.__file__),'_inference_recovery_render');m.BASE=OUT/'full';m.REPORT=REPORT
    return m

def predict():
    m=predictor();os.environ['CUDA_VISIBLE_DEVICES']='1'
    import torch
    with torch.no_grad():m.predict()
    guard();write(OUT/'recovery-complete.json',dict(plan_sha256=sha(PLAN),prediction_commit_sha256=sha(m.BASE/'prediction-commit.json'),original_failure_preserved=True))

def score():
    m=predictor();c=read(OUT/'recovery-complete.json')
    if c['plan_sha256']!=sha(PLAN)or c['prediction_commit_sha256']!=sha(m.BASE/'prediction-commit.json'):raise ValueError('Complete recovered predictions required')
    scorer=clone(ROOT/'scripts/compositional_dynamics_v17/matched_appearance_score.py','_recovered_score');scorer.r=m;scorer.score();guard()

def video():
    m=predictor();v=clone(ROOT/'scripts/compositional_dynamics_v17_reporting/matched_video.py','_recovered_video');v.BASE=m.BASE;v.REPORT=REPORT;v.build();guard()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','smoke','predict','score','video']);args=a.parse_args();globals()[args.stage]()
