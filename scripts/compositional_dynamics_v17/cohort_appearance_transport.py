"""Additive cohort transport, copied frozen development renderer; no reference reads."""
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

def render_query(p,output):
    BASE=Path(output)
    if BASE.exists():raise ValueError('Preserve predicted image arrays')
    BASE.mkdir(parents=True);sys.path.insert(0,str(AUTHOR));os.environ['CUDA_VISIBLE_DEVICES']='1'
    import torch
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gs_render import remove_gaussians_with_low_opacity
    from scripts.compositional_neural_mechanics.mechanics import Q2Mechanics,GeometrySpec
    pc=GaussianModel(3);pc.isotropic=True;pc.load_ply(str(ROOT/p['checkpoint']));pc=remove_gaussians_with_low_opacity(pc)
    if len(pc.get_xyz)==0:raise ValueError('No visible-opacity Gaussian remains')
    if 'native' in p['methods']:
        with np.load(ROOT/p['native_directory']/p['native_array'])as f:points=f['positions_m'];native_times=f['times_s']
        if not np.allclose(native_times,p['times_s'],atol=1e-12,rtol=0):raise ValueError('Native clock mismatch')
    if 'shared_q2' in p['methods']:
        with np.load(ROOT/p['ours_directory']/'trajectory.npz')as f:
            idx=np.arange(0,len(f['times_s']),10);times=f['times_s'][idx];u=f['u'][idx];rest=f['rest_nodes']
        if not np.allclose(times,p['times_s'],atol=1e-12,rtol=0):raise ValueError('Shared clock mismatch')
        model=Q2Mechanics(geometry=GeometrySpec(dimensions=tuple(p['geometry_dimensions_m'])),n=p['q2_n'],order=3,device='cpu')
        if not np.allclose(model.rest,rest,atol=1e-12,rtol=0):raise ValueError('Q2 saved node ordering mismatch')
    with torch.no_grad():
        xyz=pc.get_xyz.detach().cpu().numpy();scales=pc.get_scaling.detach().cpu().numpy();quat=pc.get_rotation.detach().cpu().numpy();R=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();cov=np.einsum('nij,nj,nkj->nik',R,scales**2,R)
        native=native_states(points,pc.get_xyz,pc.get_features_dc.squeeze(1),pc.get_rotation,pc.get_opacity) if 'native' in p['methods'] else None
        shared=q2_states(model,u,xyz,cov) if 'shared_q2' in p['methods'] else None
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
    if len(p['methods'])==2:
        for camera in p['cameras']:
            with np.load(BASE/'predictions/native'/f"000-{camera['name']}.npz")as a,np.load(BASE/'predictions/shared_q2'/f"000-{camera['name']}.npz")as b:
                initial_errors.append(float(np.max(abs(a['rgb']-b['rgb']))))
        if max(initial_errors)>1e-4:raise ValueError('Shared initial appearance mismatch')
    return dict(records=records,gaussians=len(pc.get_xyz),initial_rgb_max_difference=max(initial_errors)if initial_errors else None)
