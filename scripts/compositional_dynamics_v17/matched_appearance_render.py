"""Shared initial learned appearance, native versus Q2 transport. References scored last."""
import os,sys,json,time,copy
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import matched_appearance_native_resolution as fit
from scripts.compositional_dynamics_v17.matched_gaussian_states import q2_states,native_states
ROOT=fit.ROOT;AUTHOR=fit.AUTHOR;REPORT=fit.REPORT
BASE=ROOT/'runs/compositional-dynamics-v17/matched-appearance-render'
PLAN=REPORT/'dynamic-plan.json'
NATIVE=ROOT/'runs/compositional-dynamics-v17/native-phystwin-graph-recovery'
OURS=ROOT/'runs/compositional-dynamics-v17/matched-material-fine-forecast'
sha=fit.sha;read=fit.read;write=fit.write

def prepare(ours_directory=None,ours_plan=None,native_directory=None,native_array='forecast.npz',native_commit='recovery-complete.json'):
    native_directory=Path(native_directory)if native_directory else NATIVE
    if (native_array,native_commit)not in [('forecast.npz','recovery-complete.json'),('prediction.npz','complete.json')]:raise ValueError('Explicit supported native commit contract')
    ours_directory=Path(ours_directory)if ours_directory else OURS
    ours_plan=Path(ours_plan)if ours_plan else ROOT/'reports/compositional-dynamics-v17/matched-material-fine-forecast-plan.json'
    specification=read(ours_plan)
    coarse_commit=ROOT/'runs/compositional-dynamics-v17/matched-material-forecast/prediction-commit.json'
    if not read(coarse_commit)['completed']:raise ValueError('Coarse and fine prediction commits required before image evaluation')
    if PLAN.exists()or BASE.exists():raise ValueError('Preserve prepared rendering')
    fit.guard();appearance=read(fit.BASE/'training-complete.json');native=read(native_directory/native_commit);ours=read(ours_directory/'prediction-commit.json')
    if not appearance['complete']or not native['completed']or not ours['completed']:raise ValueError('Both physics and valid appearance commits required')
    if appearance['gaussian_count']<=0:raise ValueError('Positive learned Gaussian count')
    required={ROOT/appearance['checkpoint']:appearance['checkpoint_sha256'],native_directory/native_array:native['forecast_sha256'if native_array=='forecast.npz'else'prediction_sha256'],ours_directory/'trajectory.npz':ours['trajectory_sha256']}
    if any(sha(p)!=h for p,h in required.items()):raise ValueError('Committed artifact changed')
    from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure
    sources=closure([Path(__file__),ROOT/'scripts/compositional_dynamics_v17/matched_appearance_score.py']);sources+=list(AUTHOR.rglob('*.py'))
    sources += [ROOT/name for name in ['scripts/compositional_dynamics_v17/native_binding_smoke.py','tests/test_matched_gaussian_states_v17.py','tests/test_matched_appearance_score_v17.py']]
    inputs=list(required)+[ROOT/'reports/compositional-dynamics-v17/native-gaussian-binding-smoke.json',fit.PLAN,fit.BASE/'training-complete.json',native_directory/native_commit,ours_directory/'prediction-commit.json',ours_plan,coarse_commit]
    p=dict(native_directory=str(native_directory.relative_to(ROOT)),native_array=native_array,native_commit=native_commit,ours_directory=str(ours_directory.relative_to(ROOT)),ours_forecast_plan=str(ours_plan.relative_to(ROOT)),q2_n=specification['n'],geometry_dimensions_m=specification['geometry_dimensions_m'],scope='One development object; same learned initial GS checkpoint and same renderer. Native author K16 motion binding versus Q2 material/covariance transport; not full native reconstruction replication.',cameras=read(fit.DESIGN)['cameras'],times_s=(np.arange(26)*.01).tolist(),methods=['native','shared_q2'],checkpoint=appearance['checkpoint'],opacity_rule='Unchanged author remove_gaussians_with_low_opacity threshold0.1 applied once to BOTH initial appearances',background='Body-only rendering, common245/255 background; physical floor not drawn for either prediction/reference. Physical floor remains part of dynamics.',initial_equivalence_gate='All5 initial views RGB maximum difference<=1e-4 between two parameterizations of identical learned Gaussian state; no reference needed.',radiance='All SH coefficients and opacity frozen; native rotations/scales as author, ours F Sigma F^T. No image-space corrections.',source_sha256={str(p.relative_to(ROOT)):sha(p)for p in sources},input_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs},target_seal='All260 predicted arrays committed before this helper reads private reference geometry or generates held reference images.',failure='Any failed transport stops with partial files retained and no complete prediction commit.',runtime='render-env,physicalGPU1; Q2 used only as interpolation operator, no forward solves')
    write(PLAN,p)

def guard():
    p=read(PLAN)
    for group in ['source_sha256','input_sha256']:
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Pinned rendering input/source changed '+name)
    return p

def predict():
    p=guard()
    if BASE.exists():raise ValueError('Preserve predicted image arrays')
    BASE.mkdir();sys.path.insert(0,str(AUTHOR));os.environ['CUDA_VISIBLE_DEVICES']='1'
    import torch
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gs_render import remove_gaussians_with_low_opacity
    from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
    pc=GaussianModel(3);pc.isotropic=True;pc.load_ply(str(ROOT/p['checkpoint']));pc=remove_gaussians_with_low_opacity(pc)
    if len(pc.get_xyz)==0:raise ValueError('No visible-opacity Gaussian remains')
    with np.load(ROOT/p['native_directory']/p['native_array'])as f:points=f['positions_m'];native_times=f['times_s']
    with np.load(ROOT/p['ours_directory']/'trajectory.npz')as f:
        idx=np.arange(0,len(f['times_s']),10);times=f['times_s'][idx];u=f['u'][idx];rest=f['rest_nodes']
    if not np.allclose(times,p['times_s'],atol=1e-12,rtol=0)or not np.allclose(native_times,p['times_s'],atol=1e-12,rtol=0):raise ValueError('Exact common26frame clock required')
    model=Q2Mechanics(geometry=GeometrySpec(dimensions=tuple(p['geometry_dimensions_m'])),n=p['q2_n'],order=3,device='cpu')
    if not np.allclose(model.rest,rest,atol=1e-12,rtol=0):raise ValueError('Q2 saved node ordering mismatch')
    with torch.no_grad():
        xyz=pc.get_xyz.detach().cpu().numpy();scales=pc.get_scaling.detach().cpu().numpy();quat=pc.get_rotation.detach().cpu().numpy();R=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();cov=np.einsum('nij,nj,nkj->nik',R,scales**2,R)
        native=native_states(points,pc.get_xyz,pc.get_features_dc.squeeze(1),pc.get_rotation,pc.get_opacity)
        shared=q2_states(model,u,xyz,cov)
        records=[]
        for method in p['methods']:
            for i,t in enumerate(p['times_s']):
                current=copy.deepcopy(pc)
                if method=='native':
                    current._xyz=torch.as_tensor(native['centers_m'][i],device='cuda');current._rotation=torch.as_tensor(native['quaternions_wxyz'][i],device='cuda')
                else:
                    current.isotropic=False;current._xyz=torch.tensor(shared[i]['centers_m'],dtype=torch.float32,device='cuda');current._scaling=torch.tensor(np.log(shared[i]['scales_m']),dtype=torch.float32,device='cuda');current._rotation=torch.tensor(shared[i]['quaternions_wxyz'],dtype=torch.float32,device='cuda')
                for camera in p['cameras']:
                    K,c2w=fit.helper.calibration(camera);view=SimpleNamespace(K=torch.tensor(K,dtype=torch.float32,device='cuda'),image_width=camera['width'],image_height=camera['height'],world_view_transform=torch.tensor(np.linalg.inv(c2w).T,dtype=torch.float32,device='cuda'))
                    result=render(view,current,None,torch.full((3,),245/255,device='cuda'));rgba=result['render'].cpu().numpy();depth=result['depth'].cpu().numpy()
                    if not np.isfinite(rgba).all()or not np.isfinite(depth).all():raise ValueError('Nonfinite rendered state')
                    mask=rgba[3]>=.5;rgb=rgba[:3].transpose(1,2,0);path=BASE/'predictions'/method/f"{i:03d}-{camera['name']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,rgb=np.clip(rgb,0,1),alpha=rgba[3],body_visible=mask,body_visible_depth=np.where(mask,depth,np.inf),time_s=t)
                    records.append(dict(method=method,frame=i,time_s=t,camera=camera['name'],role=camera['role'],path=str(path.relative_to(ROOT)),sha256=sha(path),clipped_channels=int(((rgb<0)|(rgb>1)).sum())))
    initial_errors=[]
    for camera in p['cameras']:
        with np.load(BASE/'predictions/native'/f"000-{camera['name']}.npz")as a,np.load(BASE/'predictions/shared_q2'/f"000-{camera['name']}.npz")as b:
            initial_errors.append(float(np.max(abs(a['rgb']-b['rgb']))))
    if max(initial_errors)>1e-4:raise ValueError('Shared initial appearance mismatch beyond declared float tolerance')
    guard();write(BASE/'prediction-commit.json',dict(plan_sha256=sha(PLAN),records=records,expected=260,gaussians=len(pc.get_xyz),initial_rgb_max_difference=max(initial_errors),reference_images_read=False))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','predict']);a.add_argument('--ours-directory',type=Path);a.add_argument('--ours-plan',type=Path);a.add_argument('--native-directory',type=Path);a.add_argument('--native-array',default='forecast.npz');a.add_argument('--native-commit',default='recovery-complete.json');args=a.parse_args()
    prepare(args.ours_directory,args.ours_plan,args.native_directory,args.native_array,args.native_commit)if args.stage=='prepare'else predict()
